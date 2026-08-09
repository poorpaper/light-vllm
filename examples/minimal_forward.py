import torch

from light_vllm import ForwardBatch, ModelSpec, create_runner


def main() -> None:
    runner = create_runner()
    runner.load(
        ModelSpec(
            architecture="tiny-causal-lm",
            loader="init",
            model_args={"vocab_size": 128, "hidden_size": 32},
        )
    )

    batch = ForwardBatch(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))
    output = runner.forward(batch)

    print(f"generation={runner.generation}")
    print(f"logits.shape={tuple(output.logits.shape)}")


if __name__ == "__main__":
    main()
