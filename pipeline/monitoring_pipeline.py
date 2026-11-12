from src.model_monitoring import ModelMonitor


def run_monitoring_pipeline():
    monitor = ModelMonitor(
        target_column="target",
        f1_drop_threshold=0.05,
    )
    return monitor.run()


if __name__ == "__main__":
    run_monitoring_pipeline()
