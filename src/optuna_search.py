# optuna_search.py
import optuna
from train import parse_args, main as train_main
# We'll run a small function that uses the model-building code; for speed use small subset or smaller epochs

def objective(trial):
    # Suggested parameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    # For this example, you would adapt model_zoo.build... to accept dropout parameter
    # Here only pseudo-code: you should integrate configurable hyperparameters into training function
    # Return validation accuracy after short training
    val_accuracy = 0.0
    # TODO: implement a quick training run for few epochs to return validation accuracy
    return val_accuracy

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best trial:", study.best_trial.params)
