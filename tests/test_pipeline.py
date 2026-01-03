from src.train import main as train_main

def test_train_pipeline_runs():
    # This just checks that your pipeline runs without error
    train_main()
    assert True
