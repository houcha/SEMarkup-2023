from overrides import override
from copy import deepcopy

from typing import Dict, Tuple

import numpy as np

import torch
from torch import nn
from torch import Tensor, BoolTensor, LongTensor
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.nn.activations import Activation
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.training.metrics import Average, AttachmentScores
from allennlp.nn.chu_liu_edmonds import decode_mst
from allennlp.nn.util import replace_masked_values, get_range_vector, get_device_of, move_to_device


@Model.register('dependency_classifier')
class DependencyClassifier(Model):
    """
    Dozat and Manning's biaffine dependency classifier.
    I have not found any open-source implementation that can be used as a part
    of a bigger model. Mostly those are standalone parsers that take text
    as input, but we need dependency classifier to take sentence embeddings as input.
    ...so I implemented it on my own.
    It might be not 100% correct, but it does its job.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        in_dim: int, #= embedding_dim
        hid_dim: int,
        n_rel_classes_ud: int,
        n_rel_classes_eud: int,
        activation: str,
        dropout: float,
    ):
        super().__init__(vocab)

        mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hid_dim),
            Activation.by_name(activation)(),
            nn.Dropout(dropout)
        )
        self.arc_dep_mlp = deepcopy(mlp)
        self.arc_head_mlp = deepcopy(mlp)
        self.rel_dep_mlp = deepcopy(mlp)
        self.rel_head_mlp = deepcopy(mlp)

        self.arc_attention_ud = BilinearMatrixAttention(hid_dim, hid_dim, use_input_biases=False, label_dim=1)
        self.rel_attention_ud = BilinearMatrixAttention(hid_dim, hid_dim, use_input_biases=True, label_dim=n_rel_classes_ud)

        self.arc_attention_eud = BilinearMatrixAttention(hid_dim, hid_dim, use_input_biases=False, label_dim=1)
        self.rel_attention_eud = BilinearMatrixAttention(hid_dim, hid_dim, use_input_biases=True, label_dim=n_rel_classes_eud)

        # Loss.
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        # Metrics.
        self.uas_ud = Average()
        self.las_ud = Average()
        self.uas_eud = Average()
        self.las_eud = Average()

    def forward(
        self,
        embeddings: Tensor,    # [batch_size, seq_len, embedding_dim]
        deprel_labels: Tensor, # [batch_size, seq_len, seq_len]
        deps_labels: Tensor,   # [batch_size, seq_len, seq_len]
        mask_ud: Tensor,       # [batch_size, seq_len]
        mask_eud: Tensor       # [batch_size, seq_len]
    ) -> Dict[str, Tensor]:

        # [batch_size, seq_len, hid_dim]
        h_arc_head = self.arc_head_mlp(embeddings)
        h_arc_dep = self.arc_dep_mlp(embeddings)
        h_rel_head = self.rel_head_mlp(embeddings)
        h_rel_dep = self.rel_dep_mlp(embeddings)

        s_arc_ud, s_rel_ud = self._forward(
            h_arc_head, h_arc_dep, h_rel_head, h_rel_dep,
            self.arc_attention_ud, self.rel_attention_ud, mask_ud
        )
        s_arc_eud, s_rel_eud = self._forward(
            h_arc_head, h_arc_dep, h_rel_head, h_rel_dep,
            self.arc_attention_eud, self.rel_attention_eud, mask_eud
        )

        pred_arcs_ud, pred_rels_ud = self.decode_ud(s_arc_ud, s_rel_ud, mask_ud)
        pred_arcs_eud, pred_rels_eud = self.decode_eud(s_arc_eud, s_rel_eud, mask_eud)

        arc_loss_ud, rel_loss_ud = torch.tensor(0.), torch.tensor(0.)
        if deprel_labels is not None:
            arc_loss_ud, rel_loss_ud = self.loss(s_arc_ud, s_rel_ud, deprel_labels, is_multilabel=False)
            uas_ud, las_ud = self.calc_metric(pred_arcs_ud, pred_rels_ud, deprel_labels)
            self.uas_ud(uas_ud)
            self.las_ud(las_ud)

        arc_loss_eud, rel_loss_eud = torch.tensor(0.), torch.tensor(0.)
        if deps_labels is not None:
            arc_loss_eud, rel_loss_eud = self.loss(s_arc_eud, s_rel_eud, deps_labels, is_multilabel=True)
            uas_eud, las_eud = self.calc_metric(pred_arcs_eud, pred_rels_eud, deps_labels)
            self.uas_eud(uas_eud)
            self.las_eud(las_eud)

        pred_edges_ud = self._extract_nonzero_edges(pred_arcs_ud, pred_rels_ud)
        pred_edges_eud = self._extract_nonzero_edges(pred_arcs_eud, pred_rels_eud)

        return {
            'syntax_ud': pred_edges_ud,
            'syntax_eud': pred_edges_eud,
            'arc_loss_ud': arc_loss_ud,
            'rel_loss_ud': rel_loss_ud,
            'arc_loss_eud': arc_loss_eud,
            'rel_loss_eud': rel_loss_eud,
        }

    @staticmethod
    def _forward(
        h_arc_head,
        h_arc_dep,
        h_rel_head,
        h_rel_dep,
        arc_attention,
        rel_attention,
        mask
    ):
        # [batch_size, seq_len, seq_len]
        # s_arc[:, i, j] = Score of edge j -> i.
        s_arc = arc_attention(h_arc_head, h_arc_dep)
        # Mask undesirable values with -inf,
        s_arc = replace_masked_values(s_arc, DependencyClassifier._mirror_mask(mask), replace_with=-float("inf"))
        # [batch_size, seq_len, seq_len, num_labels]
        s_rel = rel_attention(h_rel_head, h_rel_dep).permute(0, 2, 3, 1)
        return s_arc, s_rel

    def decode_ud(
        self,
        s_arc: Tensor,      # [batch_size, seq_len, seq_len]
        s_rel: Tensor,      # [batch_size, seq_len, seq_len, num_labels]
        mask: Tensor,       # [batch_size, seq_len]
    ) -> Tuple[Tensor, Tensor]:

        if self.training:
            arcs, rels = self.greedy_decode_softmax(s_arc, s_rel)
        else:
            arcs, rels = self.mst_decode(s_arc, s_rel, mask)

        # Convert sequence of heads (shape [batch_size, seq_len]) to arc matrix (shape [batch_size, seq_len, seq_len]).
        # [batch_size, seq_len, seq_len]
        pred_arcs = torch.zeros_like(s_arc, dtype=torch.long).scatter(-1, arcs.unsqueeze(-1), 1)

        # Convert sequence of rels (shape [batch_size, seq_len]) to rel matrix (shape [batch_size, seq_len, seq_len]).
        # [batch_size, seq_len, seq_len]
        pred_rels = pred_arcs.clone()
        pred_rels[pred_arcs == 1] = rels.flatten()

        # Zero out padding.
        mask2d = self._mirror_mask(mask)
        pred_arcs = pred_arcs * mask2d
        pred_rels = pred_rels * mask2d

        return pred_arcs, pred_rels

    def decode_eud(
        self,
        s_arc: Tensor,      # [batch_size, seq_len, seq_len]
        s_rel: Tensor,      # [batch_size, seq_len, seq_len, num_labels]
        mask: Tensor,       # [batch_size, seq_len]
    ) -> Tuple[Tensor, Tensor]:

        pred_arcs, pred_rels = self.greedy_decode_sigmoid(s_arc, s_rel)

        # Zero out padding.
        mask2d = self._mirror_mask(mask)
        pred_arcs = pred_arcs * mask2d
        pred_rels = pred_rels * mask2d

        return pred_arcs, pred_rels

    def greedy_decode_softmax(
        self,
        s_arc: Tensor,  # [batch_size, seq_len, seq_len]
        s_rel: Tensor,  # [batch_size, seq_len, seq_len, num_labels]
    ) -> Tuple[Tensor, Tensor]:

        # Select the most probable arcs.
        # [batch_size, seq_len]
        pred_arcs = s_arc.argmax(-1)

        # Select the most probable rels for each arc.
        # [batch_size, seq_len, seq_len]
        pred_rels = s_rel.argmax(-1)
        # Select rels towards predicted arcs.
        # [batch_size, seq_len]
        pred_rels = pred_rels.gather(-1, pred_arcs[:, :, None]).squeeze(-1)

        return pred_arcs, pred_rels

    def greedy_decode_sigmoid(
        self,
        s_arc: Tensor,  # [batch_size, seq_len, seq_len]
        s_rel: Tensor,  # [batch_size, seq_len, seq_len, num_labels]
    ) -> Tuple[Tensor, Tensor]:

        # Select all probable arcs.
        # [batch_size, seq_len, seq_len]
        pred_arcs = torch.sigmoid(s_arc).round().long()

        # Select the most probable rel for each arc.
        # [batch_size, seq_len, seq_len]
        pred_rels = s_rel.argmax(-1)

        return pred_arcs, pred_rels

    def mst_decode(
        self,
        s_arc: Tensor, # [batch_size, seq_len, seq_len]
        s_rel: Tensor, # [batch_size, seq_len, seq_len, num_labels]
        mask: Tensor   # [batch_size, seq_len]
    ) -> Tuple[Tensor, Tensor]:

        batch_size, seq_len, _ = s_arc.shape

        assert get_device_of(s_arc) == get_device_of(s_rel)
        device = get_device_of(s_arc)
        s_arc = s_arc.cpu()
        s_rel = s_rel.cpu()

        # It is the most tricky part of dependency classifier.
        # If you want to get into it, first visit
        # https://docs.allennlp.org/main/api/nn/chu_liu_edmonds
        # It is not that detailed, so you better look into the source.

        # First, normalize values, as decode_mst expects values to be non-negative.
        s_arc_probs = nn.functional.softmax(s_arc, dim=-1)

        # Next, recall the hack: we use diagonal to store ROOT relation, so
        #
        # s_arc[i,j] = "Probability that j is the head of i" if i != j else "Probability that i is ROOT".
        #
        # However, allennlp's decode_mst defines 'energy' matrix as follows:
        #
        # energy[i,j] = "Score that i is the head of j",
        #
        # which means we have to transpose s_arc at first, so that:
        #
        # s_arc[i,j] = "Score that i is the head of j" if i != j else "Score that i is ROOT".
        #
        s_arc_probs_inv = s_arc_probs.transpose(1, 2)

        # Also note that decode_mst can't handle loops, as it zeroes diagonal out.
        # So, s_arc now:
        #
        # s_arc[i,j] = "Score that i is the head of j".
        #
        # However, decode_mst can produce a tree where root node
        # has a parent, since it knows nothing about root yet.
        # That is, we have to choose the latter explicitly.
        # [batch_size]
        root_idxs = s_arc_probs_inv.diagonal(dim1=1, dim2=2).argmax(dim=-1)

        predicted_arcs = []
        for batch_idx, root_idx in enumerate(root_idxs):
            # Now zero s_arc[i, root_idx] = "Probability that i is the head of root_idx" out.
            energy = s_arc_probs_inv[batch_idx]
            energy[:, root_idx] = 0.0
            # Finally, we are ready to call decode_mst,
            # Note s_arc don't know anything about labels, so we set has_labels=False.
            lengths = mask[batch_idx].sum()
            heads, _ = decode_mst(energy, lengths, has_labels=False)

            # Some vertices may be isolated. Pick heads greedily in this case.
            heads[heads <= 0] = s_arc[batch_idx].argmax(-1)[heads <= 0]
            predicted_arcs.append(heads)

        # [batch_size, seq_len]
        predicted_arcs = torch.from_numpy(np.stack(predicted_arcs)).to(torch.int64)

        # ...and predict relations.
        predicted_rels = s_rel.argmax(-1)

        # Select rels towards predicted arcs.
        # [batch_size, seq_len]
        predicted_rels = predicted_rels.gather(-1, predicted_arcs[:, :, None]).squeeze(-1)

        predicted_arcs = move_to_device(predicted_arcs, device)
        predicted_rels = move_to_device(predicted_rels, device)

        return predicted_arcs, predicted_rels

    def loss(
        self,
        s_arc: Tensor,  # [batch_size, seq_len, seq_len]
        s_rel: Tensor,  # [batch_size, seq_len, seq_len, num_labels]
        target: Tensor, # [batch_size, seq_len, seq_len]
        is_multilabel: bool
    ) -> Tuple[Tensor, Tensor]:

        # has_arc_mask[:, i, j] = True iff target has edge at index [i, j].
        # [batch_size, seq_len, seq_len]
        has_arc_mask = target != -1
        # [batch_size, seq_len]
        mask = torch.any(has_arc_mask == True, dim=-1)
        # [batch_size, seq_len, seq_len]
        mask2d = self._mirror_mask(mask)

        if is_multilabel:
            # [batch_size, seq_len]
            arc_losses = self.bce_loss(s_arc[mask], has_arc_mask[mask].float())
            arc_loss = arc_losses[mask2d[mask]].mean()
        else:
            arc_loss = self.cross_entropy(s_arc[mask], has_arc_mask[mask].long().argmax(-1))

        rel_loss = self.cross_entropy(s_rel[has_arc_mask], target[has_arc_mask])

        assert arc_loss != float("inf")
        assert rel_loss != float("inf")
        return arc_loss, rel_loss

    def calc_metric(
        self,
        pred_arcs: LongTensor,  # [batch_size, seq_len, seq_len]
        pred_rels: LongTensor,  # [batch_size, seq_len, seq_len]
        target: LongTensor,     # [batch_size, seq_len, seq_len]
    ):
        # [batch_size, seq_len, seq_len]
        has_arc_mask = target != -1
        # [batch_size, seq_len]
        mask = torch.any(has_arc_mask == True, dim=-1)
        # [batch_size, seq_len, seq_len]
        mask2d = self._mirror_mask(mask)
        
        # Multi-UAS.
        target_arcs_idxs = has_arc_mask.nonzero()
        pred_arcs_idxs = (pred_arcs * mask2d).nonzero()
        uas = self._multilabel_attachment_score(pred_arcs_idxs, target_arcs_idxs)

        # Multi-LAS.
        target_rels_idxs = (target * has_arc_mask).nonzero()
        pred_rels_idxs = (pred_rels * pred_arcs * mask2d).nonzero()
        las = self._multilabel_attachment_score(pred_rels_idxs, target_rels_idxs)

        return uas, las

    @override
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        uas_ud = self.uas_ud.get_metric(reset)
        las_ud = self.las_ud.get_metric(reset)
        uas_eud = self.uas_eud.get_metric(reset)
        las_eud = self.las_eud.get_metric(reset)
        return {
            "UD-UAS": uas_ud,
            "UD-LAS": las_ud,
            "EUD-UAS": uas_eud,
            "EUD-LAS": las_eud,
        }

    ### Private methods ###

    @staticmethod
    def _mirror_mask(mask1d: BoolTensor) -> BoolTensor:
        """
        Mirror mask symmetrically, so that mask2d[:, i, j] = True iff mask1d[:, i] = mask1d[:, j] = True.
        Example:
        >>> mask1d = torch.BoolTensor([[True, False, True, True]])
        >>> _mirror_mask(mask)
        tensor([[[ True,  True, False,  True],
                 [ True,  True, False,  True],
                 [False, False, False, False],
                 [ True,  True, False,  True]],

                [[False, False, False, False],
                 [False,  True,  True, False],
                 [False,  True,  True, False],
                 [False, False, False, False]]])
        """
        return mask1d[:, None, :] * mask1d[:, :, None]

    @staticmethod
    def _multilabel_attachment_score(
        pred_labels: LongTensor,
        true_labels: LongTensor
    ) -> float:
        """
        Measures similarity of sets of indices.
        Basically an Intersection over Union measure, but "Union" is replaced with "Max", so that
        this score is equal to UAS/LAS for single-label classification.
        """
        # Fisrt convert tensors to lists of tuples.
        pred_labels = [tuple(index) for index in pred_labels.tolist()]
        true_labels = [tuple(index) for index in true_labels.tolist()]
        # Then calculate IoU.
        intersection = set(pred_labels) & set(true_labels)
        union = set(pred_labels) | set(true_labels)
        max_len = len(pred_labels) if len(pred_labels) > len(true_labels) else len(true_labels)
        return len(intersection) / max_len

    @staticmethod
    def _extract_nonzero_edges(
        arcs: LongTensor, # [batch_size, seq_len, seq_len]
        rels: LongTensor  # [batch_size, seq_len, seq_len]
    ) -> LongTensor:
        """
        Aggregate arcs and relations into single array of tuples (batch_index, edge_to, edge_from, rel_id).
        Works for both UD and E-UD matrices.
        """
        assert len(arcs.shape) == 3
        assert arcs.shape == rels.shape

        # [batch_size, seq_len, seq_len, num_classes]
        rels_one_hot = F.one_hot(rels)
        nonzero_arcs_positions = arcs.nonzero(as_tuple=True)
        # Zero out all one-hots but ones present in nonzero_arcs_positions.
        rels_filtered = torch.zeros_like(rels_one_hot)
        rels_filtered[nonzero_arcs_positions] = rels_one_hot[nonzero_arcs_positions]
        # Now rels_filtered has one-hot vector at [i, j, k] position iff i-th batch has (j, k) arc.
        return rels_filtered.nonzero()
        
