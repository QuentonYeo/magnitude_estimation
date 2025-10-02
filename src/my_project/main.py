from my_project.tutorial.tutorial import (
    test_load_data,
    test_generator,
    train_phasenet,
    evaluate_phasenet,
)

if __name__ == "__main__":
    # test_load_data()
    # test_generator()
    # train_phasenet("PhaseNet_ETHZ", learning_rate=1e-2, epochs=10)

    # run with: uv run src/my_project/main.py --model_path <root/path-to-model>
    evaluate_phasenet()
