from mlflow.tracking import MlflowClient

client = MlflowClient()

experiment = client.get_experiment_by_name(
    "Teen_Mental_Health"
)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)

run_id = runs[0].info.run_id

with open("run_id.txt", "w") as f:
    f.write(run_id)

print(run_id)