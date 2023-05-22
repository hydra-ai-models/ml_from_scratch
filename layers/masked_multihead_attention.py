'''
    Class implementing masked multihead attention.
'''

import torch, torch.nn as nn

class MaskedMultiHeadAttention(nn.Module):
    '''
        Multihead attention layer which can perform causal or non causal attention.
    '''
    def __init__(self, mask: bool, num_heads: int, head_dimension: int, input_dimension: int):
        '''
            Args:
                1. mask - Whether to apply causal attention or not.
                2. num_heads - Number of heads to use when computing multihead attention.
                3. head_dimension - Dimension of the head embedding.
                4. input_dimension - dimension of the input embedding.
        '''
        super().__init__()
        self.mask: bool = mask
        self.num_heads = num_heads
        self.head_dimension = head_dimension
        self.Wq = nn.Linear(input_dimension, num_heads * head_dimension) # (Re, nh * Rh)
        self.Wk = nn.Linear(input_dimension, num_heads * head_dimension)
        self.Wv = nn.Linear(input_dimension, num_heads * head_dimension)
        self.head_merge_layer = nn.Linear(num_heads*head_dimension, input_dimension)

    def forward(self, queries, keys, values):
        # queries - (B, T, input_dimension)
        (batch_size, block_size, _) = queries.shape
        projected_queries = self.Wq(queries) # (B, T, nh*Rh)
        reshaped_queries = projected_queries.view(-1, block_size, self.num_heads, self.head_dimension) # (B, T, nh, Rh)
        transposed_queries = reshaped_queries.transpose(1, 2) # (B, nh, T, Rh)\

        projected_keys = self.Wk(keys) # (B, T, nh*Rh)
        reshaped_keys = projected_keys.view(-1, block_size, self.num_heads, self.head_dimension)
        transposed_keys = reshaped_keys.transpose(1, 2) # (B, nh, T, Rh)

        projected_values = self.Wv(values) # (B, T, nh*Rh)
        reshaped_values = projected_values.view(-1, block_size, self.num_heads, self.head_dimension)
        transposed_values = reshaped_values.transpose(1,2) # (B, nh, T, Rh)


        attention = transposed_queries @ (transposed_keys.transpose(2, 3)) # (B, nh, T, T)
        if self.mask:
            tril_mat = torch.tril(torch.ones(block_size, block_size))
            attention.masked_fill(tril_mat == 0, -torch.inf)
        attention = nn.functional.softmax(attention, dim = 2) / self.head_dimension**(0.5) # (B, nh, T, T)
        attention_output_multihead = attention @ transposed_values # (B, nh, T, Rh)

        # Combining multiple heads.
        transposed_attention_output_multihead = attention_output_multihead.transpose(1, 2) # (B, T, nh, Rh)
        rehaped_attention_output_multihead = transposed_attention_output_multihead.contiguous().view(-1, block_size, self.num_heads * self.head_dimension)
        attention_output = self.head_merge_layer(rehaped_attention_output_multihead)
        return attention_output

class MaskedMultiHeadAttentionWithLoRAFinetuning(MaskedMultiHeadAttention):
    '''
        Multihead attention layer which can perform causal or non causal attention and support
        LoRA finetuning (https://arxiv.org/abs/2106.09685).
    '''
    def __init__(self, mask: bool, num_heads: int, head_dimension: int, input_dimension: int, mode: str, lora_rank: int = 4):
        '''
            Args:
                1. mask - See documentation of parent class (MaskedMultiHeadAttention).
                2. num_heads - See documentation of parent class (MaskedMultiHeadAttention).
                3. head_dimension - See documentation of parent class (MaskedMultiHeadAttention).
                4. input_dimension - See documentation of parent class (MaskedMultiHeadAttention).
                5. mode - Whether doing pretraining or finetuning. Supported values are ['pretraining', 'finetuning']
                6. lora_rank - Rank to be used for the LoRA matrices A and B. Default value is 4.
        '''
        super().__init__(mask, num_heads, head_dimension, input_dimension)

        self.supported_mode_values: list[str] = ['pretraining', 'finetuning']
        assert mode in supported_mode_values, f'Unsupported value {mode}. Supported values are {self.supported_mode_values}'
        self.mode = mode

        if mode == 'finetuning':
            self.lora_Aq = nn.Linear(input_dimension, lora_rank)
            self.lora_Bq = nn.Linear(lora_rank, num_heads * head_dimension)
            self.lora_Ak = nn.Linear(input_dimension, lora_rank)
            self.lora_Bk = nn.Linear(lora_rank, num_heads * head_dimension)
            self.lora_Av = nn.Linear(input_dimension, lora_rank)
            self.lora_Bv = nn.Linear(lora_rank, num_heads * head_dimension)

        def forward(self, queries, keys, values):
            # queries - (B, T, input_dimension)
            (batch_size, block_size, _) = queries.shape
            projected_queries = self.Wq(queries) # (B, T, nh*Rh)
            if mode == finetuning:
                projected_queries += self.Bq(self.Aq(queries)) # (B, T, nh*Rh)
            reshaped_queries = projected_queries.view(-1, block_size, self.num_heads, self.head_dimension) # (B, T, nh, Rh)
            transposed_queries = reshaped_queries.transpose(1, 2) # (B, nh, T, Rh)\

            projected_keys = self.Wk(keys) # (B, T, nh*Rh)
            if mode == finetuning:
                projected_keys += self.Bk(self.Ak(keys)) # (B, T, nh*Rh)
            reshaped_keys = projected_keys.view(-1, block_size, self.num_heads, self.head_dimension)
            transposed_keys = reshaped_keys.transpose(1, 2) # (B, nh, T, Rh)

            projected_values = self.Wv(values) # (B, T, nh*Rh)
            if mode == finetuning:
                projected_values += self.Bv(self.Av(keys)) # (B, T, nh*Rh)
            reshaped_values = projected_values.view(-1, block_size, self.num_heads, self.head_dimension)
            transposed_values = reshaped_values.transpose(1,2) # (B, nh, T, Rh)


            attention = transposed_queries @ (transposed_keys.transpose(2, 3)) # (B, nh, T, T)
            if self.mask:
                tril_mat = torch.tril(torch.ones(block_size, block_size))
                attention.masked_fill(tril_mat == 0, -torch.inf)
            attention = nn.functional.softmax(attention, dim = 2) / self.head_dimension**(0.5) # (B, nh, T, T)
            attention_output_multihead = attention @ transposed_values # (B, nh, T, Rh)

            # Combining multiple heads.
            transposed_attention_output_multihead = attention_output_multihead.transpose(1, 2) # (B, T, nh, Rh)
            rehaped_attention_output_multihead = transposed_attention_output_multihead.contiguous().view(-1, block_size, self.num_heads * self.head_dimension)
            attention_output = self.head_merge_layer(rehaped_attention_output_multihead)
            return attention_output
