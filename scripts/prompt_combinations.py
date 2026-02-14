import math
import re

import gradio as gr
import modules.scripts as scripts

from modules.processing import fix_seed, process_images


def _expand_curly_blocks(prompt):
    """Expand {a|b} blocks into all combinations."""
    match = re.search(r"\{([^{}]*)\}", prompt)
    if match is None:
        return [prompt]

    start, end = match.span()
    options = [option.strip() for option in match.group(1).split("|")]

    expanded = []
    for option in options:
        next_prompt = prompt[:start] + option + prompt[end:]
        expanded.extend(_expand_curly_blocks(next_prompt))

    return expanded


class Script(scripts.Script):
    def title(self):
        return "Curly brace combinations"

    def ui(self, is_img2img):
        with gr.Row():
            different_seeds = gr.Checkbox(
                label="Use different seed for each picture",
                value=False,
                elem_id=self.elem_id("different_seeds"),
            )
            prompt_type = gr.Radio(
                ["positive", "negative"],
                label="Select prompt",
                elem_id=self.elem_id("prompt_type"),
                value="positive",
            )

        return [different_seeds, prompt_type]

    def run(self, p, different_seeds, prompt_type):
        fix_seed(p)

        if prompt_type not in ["positive", "negative"]:
            raise ValueError(f"Unknown prompt type {prompt_type}")

        source_prompt = p.prompt if prompt_type == "positive" else p.negative_prompt
        source_prompt = source_prompt[0] if isinstance(source_prompt, list) else source_prompt

        all_prompts = _expand_curly_blocks(source_prompt)

        p.n_iter = math.ceil(len(all_prompts) / p.batch_size)
        p.do_not_save_grid = True

        print(f"Curly brace combinations will create {len(all_prompts)} images using a total of {p.n_iter} batches.")

        if prompt_type == "positive":
            p.prompt = all_prompts
        else:
            p.negative_prompt = all_prompts

        p.seed = [p.seed + (i if different_seeds else 0) for i in range(len(all_prompts))]

        return process_images(p)
