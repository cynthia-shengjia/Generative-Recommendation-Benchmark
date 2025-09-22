import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Config
from transformers.generation.beam_search import BeamScorer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerateBeamEncoderDecoderOutput
from typing import Union, Optional, Tuple, Dict, Any


class CustomT5ForConditionalGeneration(T5ForConditionalGeneration):
    
    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        **model_kwargs,
    ):
        """
        Custom beam search implementation for T5 encoder-decoder model
        """
        # Initialize values from generation config
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        sequential = generation_config.low_memory
        do_sample = generation_config.do_sample

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # Initialize output tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # For encoder-decoder models, retrieve encoder outputs
        encoder_attentions = None
        encoder_hidden_states = None
        if return_dict_in_generate and self.config.is_encoder_decoder:
            if "encoder_outputs" in model_kwargs:
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

        # Initialize beam scores
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False
        decoder_prompt_len = input_ids.shape[-1]  # record decoder prompt length

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # Prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            # Add output controls
            if output_attentions:
                model_inputs["output_attentions"] = True
            if output_hidden_states:
                model_inputs["output_hidden_states"] = True

            # For T5, we don't support sequential processing in this implementation
            # Always use the standard approach
            outputs = self(**model_inputs, return_dict=True)

            # Update model kwargs for next generation step
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue

            # Process next token logits
            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)
            
            # Apply logits processors
            next_token_logits_processed = logits_processor(input_ids, next_token_logits)
            next_token_scores_processed = nn.functional.log_softmax(
                next_token_logits_processed, dim=-1
            )

            # Add beam scores
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store outputs if required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if hasattr(outputs, 'decoder_attentions') else (outputs.attentions,)
                    )
                    if hasattr(outputs, 'cross_attentions'):
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,) if hasattr(outputs, 'decoder_hidden_states') else (outputs.hidden_states,)
                    )

            # Reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Beam token selection
            n_eos_tokens = len(eos_token_id) if isinstance(eos_token_id, (list, tuple)) else 1
            n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
            
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)
            else:
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
                )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # Process beam search results
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            # Clean up to save memory
            del outputs

            # Reorder past key values if they exist
            if model_kwargs.get("past_key_values") is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            cur_len += 1

            # Check stopping criteria
            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                this_peer_finished = True

        # Finalize beam search
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            return GenerateBeamEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return sequence_outputs["sequences"]

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorders the cache for beam search
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # T5 past key values are tuples of (encoder, decoder) states
            reordered_past_layer = ()
            for layer_past_component in layer_past:
                # Reorder the batch dimension of the past state
                reordered_past_layer += (layer_past_component.index_select(0, beam_idx),)
            reordered_past += (reordered_past_layer,)
        return reordered_past

    def _has_unfinished_sequences(self, this_peer_finished, synced_gpus, device):
        """
        Check if there are unfinished sequences
        """
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            return not this_peer_finished
        return True
