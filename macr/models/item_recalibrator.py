from __future__ import annotations
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from macr.losses import masked_bce_with_logits, supervised_alignment_loss
from macr.models.attention import TargetAwareAttention
from macr.models.interactions import NCDInteraction

class MACRItemRecalibrator(PreTrainedModel):

    def __init__(self, config, encoder, student_anchors: torch.Tensor, kc_embeddings: torch.Tensor, align_weight: float=0.5, use_prednet: bool=True) -> None:
        super().__init__(config)
        self.encoder = encoder
        self.hidden_size = int(config.hidden_size)
        self.align_weight = align_weight
        self.use_prednet = use_prednet
        self.register_buffer('student_anchors', student_anchors.float())
        self.kc_base = nn.Embedding.from_pretrained(kc_embeddings.float(), freeze=True)
        self.target_attention = TargetAwareAttention(self.hidden_size)
        self.head_q = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.head_beta = nn.Linear(self.hidden_size, self.hidden_size)
        self.head_alpha = nn.Linear(self.hidden_size, 1)
        self.head_beta_kc = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.head_alpha_kc = nn.Linear(self.hidden_size, 1, bias=False)
        nn.init.zeros_(self.head_beta_kc.weight)
        nn.init.zeros_(self.head_alpha_kc.weight)
        self.q_norm = nn.LayerNorm(self.hidden_size)
        self.interaction = NCDInteraction(self.hidden_size, use_prednet=use_prednet)
        for name, param in self.named_parameters():
            if 'encoder' in name and 'lora' not in name:
                param.requires_grad = False
        for param in self.interaction.parameters():
            param.requires_grad = False

    def encode_items(self, input_ids_prompt: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids_prompt, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        last_token = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
        row_index = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[row_index, last_token]

    def project_measurements(self, h_item: torch.Tensor, kc_ids: torch.Tensor, kc_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        safe_kc_ids = kc_ids.masked_fill(kc_mask == 0, 0)
        h_kcs = self.kc_base(safe_kc_ids)
        h_kc = self.target_attention(h_item, h_kcs, kc_mask)
        z = torch.cat([h_item, h_kc], dim=-1)
        q = torch.sigmoid(self.q_norm(self.head_q(z)))
        q = torch.relu(q - 0.2)
        beta = self.head_beta(h_item) + self.head_beta_kc(h_kc)
        alpha = torch.sigmoid(self.head_alpha(h_item) + self.head_alpha_kc(h_kc))
        measurement = torch.cat([q, beta, alpha], dim=-1)
        return {'q': q, 'beta': beta, 'alpha': alpha, 'measurement': measurement, 'h_item': h_item}

    def forward(self, item_id: torch.Tensor, input_ids_prompt: torch.Tensor, attention_mask: torch.Tensor, kc_ids: torch.Tensor, kc_mask: torch.Tensor, labels: torch.Tensor | None=None, teacher_q: torch.Tensor | None=None, teacher_beta: torch.Tensor | None=None, teacher_alpha: torch.Tensor | None=None, **_: object) -> dict[str, torch.Tensor | None]:
        h_item = self.encode_items(input_ids_prompt, attention_mask)
        projected = self.project_measurements(h_item, kc_ids, kc_mask)
        logits = self.interaction(self.student_anchors.to(h_item.device), projected['q'], projected['beta'], projected['alpha'])
        loss = None
        if labels is not None:
            response_loss = masked_bce_with_logits(logits, labels)
            align_loss = supervised_alignment_loss(projected['q'], projected['beta'], projected['alpha'], teacher_q, teacher_beta, teacher_alpha)
            loss = response_loss + self.align_weight * align_loss
        return {'loss': loss, 'logits': logits, 'item_id': item_id, 'h_item': projected['h_item'], 'pred_q': projected['q'], 'raw_beta': projected['beta'], 'raw_alpha': projected['alpha'], 'measurement': projected['measurement']}
