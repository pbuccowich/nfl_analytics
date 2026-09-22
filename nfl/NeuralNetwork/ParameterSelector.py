import itertools
import copy
import random
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

class BaseSearch:
    def __init__(
        self,
        param_grid: dict,
        solver_cls,
        model_cls,
        criterion,
        device: str = "cpu",
        num_epochs: int = 100,
        input_size: int = 10,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        log_dir: str = "runs/hyperparameter_search",
    ):
        """
        Base Hyperparameter Optimizer.
        
        param_grid requires lists for keys:
          'batch_size', 'lr', 'num_hidden_layers', 'hidden_size', 'weight_decay'
        """
        self.param_grid = param_grid
        self.solver_cls = solver_cls
        self.model_cls = model_cls
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.input_size = input_size
        self.betas = betas
        self.eps = eps
        self.log_dir = log_dir

    def _evaluate_config(self, params: dict, X_tr, y_tr, X_v, y_v, run_name: str) -> dict:
        """Helper to instantiate the model, optimizer, writer, and run Solver."""
        writer = SummaryWriter(log_dir=f"{self.log_dir}/{run_name}")

        model = self.model_cls(
            input_size=self.input_size,
            num_hidden_layers=int(params["num_hidden_layers"]),
            hidden_size=int(params["hidden_size"]),
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(params["lr"]),
            betas=self.betas,
            eps=self.eps,
            weight_decay=float(params["weight_decay"]),
        )

        # Cast int params to avoid scalar numpy type bugs
        hparams = {
            "batch_size": int(params["batch_size"]),
            "lr": float(params["lr"]),
            "num_hidden_layers": int(params["num_hidden_layers"]),
            "hidden_size": int(params["hidden_size"]),
            "weight_decay": float(params["weight_decay"]),
        }

        solver = self.solver_cls(
            model=model,
            device=self.device,
            num_epochs=self.num_epochs,
            batch_size=hparams["batch_size"],
            optimizer=optimizer,
            criterion=self.criterion,
            writer=writer,
            run_name=run_name,
            hparams=hparams,
        )

        metrics = solver.train(X_tr, y_tr, X_v, y_v, modelName=run_name, saveBest=True)
        writer.close()

        return {"run_name": run_name, **hparams, **metrics}


class GridSearch(BaseSearch):
    def run(self, X_tr, y_tr, X_v, y_v, metric: str = "best_val_loss") -> tuple[pd.DataFrame, dict]:
        """Exhaustively runs all parameter combinations."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(itertools.product(*values))

        results = []
        print(f"--- Starting Grid Search ({len(combinations)} total runs) ---")

        for idx, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            run_name = (
                f"grid_bs{params['batch_size']}_lr{params['lr']:.1e}_"
                f"hl{params['num_hidden_layers']}_hs{params['hidden_size']}_wd{params['weight_decay']:.1e}"
            )
            print(f"Run [{idx + 1}/{len(combinations)}]: {run_name}")

            run_dict = self._evaluate_config(params, X_tr, y_tr, X_v, y_v, run_name)
            results.append(run_dict)

        df = pd.DataFrame(results)
        best_row = df.sort_values(by=metric, ascending=True).iloc[0].to_dict()
        return df, best_row


class GeneticSearch(BaseSearch):
    def __init__(
        self,
        param_grid: dict,
        solver_cls,
        model_cls,
        criterion,
        pop_size: int = 12,
        generations: int = 5,
        mutation_rate: float = 0.2,
        top_k: int = 4,
        **kwargs,
    ):
        super().__init__(param_grid, solver_cls, model_cls, criterion, **kwargs)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.top_k = top_k

    def _random_individual(self) -> dict:
        return {k: random.choice(v) for k, v in self.param_grid.items()}

    def _crossover(self, parent1: dict, parent2: dict) -> dict:
        child = {}
        for k in self.param_grid.keys():
            child[k] = parent1[k] if random.random() < 0.5 else parent2[k]
        return child

    def _mutate(self, individual: dict) -> dict:
        mutated = copy.deepcopy(individual)
        for k, choices in self.param_grid.items():
            if random.random() < self.mutation_rate:
                mutated[k] = random.choice(choices)
        return mutated

    def run(self, X_tr, y_tr, X_v, y_v, metric: str = "best_val_loss") -> tuple[pd.DataFrame, dict]:
        """Runs the evolutionary optimization loop over generations."""
        # Initialize population
        population = [self._random_individual() for _ in range(self.pop_size)]
        evaluated_cache = {}  # Caches previously evaluated configs to avoid redundant runs
        results = []

        print(f"--- Starting Genetic Search ({self.generations} generations, pop size {self.pop_size}) ---")

        for gen in range(self.generations):
            print(f"\n=== Generation {gen + 1}/{self.generations} ===")
            gen_scores = []

            for idx, ind in enumerate(population):
                # Create a static hashable key for caching
                config_key = tuple(sorted(ind.items()))

                if config_key in evaluated_cache:
                    metrics = evaluated_cache[config_key]
                    print(f"Ind [{idx + 1}/{self.pop_size}]: Cached ({metrics[metric]:.4f})")
                else:
                    run_name = (
                        f"gen{gen + 1}_ind{idx + 1}_bs{ind['batch_size']}_lr{ind['lr']:.1e}_"
                        f"hl{ind['num_hidden_layers']}_hs{ind['hidden_size']}_wd{ind['weight_decay']:.1e}"
                    )
                    metrics = self._evaluate_config(ind, X_tr, y_tr, X_v, y_v, run_name)
                    evaluated_cache[config_key] = metrics
                    results.append(metrics)
                    print(f"Ind [{idx + 1}/{self.pop_size}]: {metric} = {metrics[metric]:.4f}")

                gen_scores.append((metrics[metric], ind))

            # Rank population by target metric (ascending = lower loss is better)
            gen_scores.sort(key=lambda x: x[0])
            survivors = [ind for _, ind in gen_scores[: self.top_k]]

            # Breed next generation if not on the final iteration
            if gen < self.generations - 1:
                next_pop = copy.deepcopy(survivors)  # Elitism: keep top-k directly
                while len(next_pop) < self.pop_size:
                    p1, p2 = random.sample(survivors, 2)
                    child = self._crossover(p1, p2)
                    child = self._mutate(child)
                    next_pop.append(child)
                population = next_pop

        df = pd.DataFrame(results)
        best_row = df.sort_values(by=metric, ascending=True).iloc[0].to_dict()
        return df, best_row