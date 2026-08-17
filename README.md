# WalletStatistics

Independent Python simulation code for the **Boltzmann Draw**, a probabilistic
coin-selection method for token-based payment systems. This implementation is
related to the paper
[Coin selection by Random Draw according to the Boltzmann distribution](https://arxiv.org/abs/2602.17490).

## Relationship to the paper and data availability

Marc Winstel is a co-author of the corresponding paper and developed this code
as an independent implementation for verifying and cross-checking its results.
This repository was not the primary codebase used to generate the numerical data
reported in the paper.

Those data were generated independently by two other co-authors using separate
codebases. Their implementations and result data are not publicly available at
this time. This repository should therefore be understood as an independent
validation implementation and research simulator, not as the paper's complete
reproduction package. It can generate its own simulation results for the models
and scenarios documented below.

## Model

The simulator represents a wallet as a set of uniquely identified, positive-value
tokens. Deposits add tokens; payments repeatedly draw tokens until the requested
amount is covered and return any excess as a change token. All monetary values are
rounded to two decimal places, so the minimum denomination is
$d_s = 0.01$.

For a token with value $e_i$, the canonical Boltzmann Draw assigns a weight

$$
w_i = \exp(-\beta e_i).
$$

Tokens are sampled according to their normalized weights. The inverse-temperature
parameter $\beta$ can be updated dynamically from the current token count $n$
and total wallet value $E$ using one of three modes:

- `legacy`: $\beta = n/E$
- `microcanonicalExact`:
  $\beta = \sum_{k=1}^{n-1} 1/(E-kd_s)$
- `microcanonicalApprox`: $\beta = (n-1)/E$ when
  $E > f(n-1)d_s$; otherwise the exact expression is used. The configurable
  approximation factor $f$ defaults to `10.0`.

Every simulation starts with one funding token worth $10^7$.

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/marc-win2/WalletStatistics.git
cd WalletStatistics
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Running a simulation

The following example runs one small Gaussian simulation using the legacy beta
adjustment and writes its results to `DemoData/` and `DemoDataGlobal/`:

```bash
python3 - <<'PY'
from main import runStandaloneSimulationExperiment

token_buckets = [10**i for i in range(-2, 10)]
runStandaloneSimulationExperiment(
    token_buckets,
    dataDirectory="DemoData",
    globalDataDirectory="DemoDataGlobal",
    numSimulations=1,
    noPayments=100,
    transactionScenario="gaussian",
    betaAdjustmentMode="legacy",
)
PY
```

The two experiment workloads are:

- `gaussian`: Gaussian payments interleaved with Gaussian deposits.
- `dirichletFloat`: groups of floating-point Dirichlet payments balanced by a
  constant deposit. Values are still rounded to the minimum denomination.

The integer multinomial/Dirichlet generator remains available in the source but
is not part of the standard experiment matrix.

Running `python3 main.py` starts the full matrix: all three beta modes crossed
with both workloads, with 100 independent runs and 100 payments per run. Change
the number of payments with `--num_iter`; the value must be positive and divisible
by 10 because the Dirichlet workload generates payments in groups of ten:

```bash
python3 main.py --num_iter 100000
```

Large iteration counts are computationally expensive. Results are written below
`Simulations/BetaAdjustmentMatrix/`, with a separate directory for each of the
six configurations.

Each configuration contains per-run data and plots in `Data/` and aggregate
summaries in `DataGlobal/`. The `.dat` files contain either one value per line or
an index/value pair, depending on the recorded quantity.

The `payment_token_count_*.dat` metric counts all tokens participating in the
payment operation: selected input tokens plus the generated change token when a
payment overdraws its inputs. It therefore may be one greater than the number of
selected input tokens.

## Tests

Run the unit test suite with:

```bash
python3 -m unittest discover -s tests -v
```

The tests use temporary directories and do not create simulation data in the
repository.

## Aggregating simulation output

`averageSimulationPlots.py` averages histories and final token values across
simulation runs. By default it reads `./Data`, writes `./DataGlobal`, aggregates
100 runs with 100,000 payments each, and processes histories in chunks of 20,000
rows:

```bash
python3 averageSimulationPlots.py
```

All settings can be changed from the command line, for example:

```bash
python3 averageSimulationPlots.py \
  --num_runs 10 \
  --num_payments 1000 \
  --chunk_size 500 \
  --data_path Simulations/example/Data \
  --save_path Simulations/example/DataGlobal
```

Chunking limits peak memory by reading and aggregating only part of every run's
history at a time. It does not change the calculated means or standard
deviations.

## Known limitation

The bucket-accelerated selection path (`useBucketsForProbabilityComp=True`) is
experimental and is not used by the standard experiments. It currently assigns
one Boltzmann weight to each non-empty bucket without accounting for how many
tokens the bucket contains. Consequently, it does not reproduce token-level
Boltzmann sampling when bucket occupancies differ. Unlike the token-level path,
it also has no fallback if every bucket weight numerically underflows to zero.
Keep `useBucketsForProbabilityComp=False` for the supported simulation setup.

## Source layout

- `simulation.py`: wallet lifecycle, payments, deposits, and beta adjustment.
- `coinselection.py`: Boltzmann weights and token-selection distributions.
- `wallet.py`: token and wallet models plus denomination handling.
- `transaction.py`: random transaction generators.
- `main.py`: experiment orchestration and output generation.
- `averageSimulationPlots.py`: post-processing of simulation output.

## Acknowledgment of AI assistance

Parts of the preparation of this repository for publication on GitHub, as well
as later-stage code support primarily concerning `main.py`, were carried out
with assistance from OpenAI's GPT-5.6 Sol model via Codex. The resulting changes
were reviewed and accepted by the repository author.

## License

This project is available under the [MIT License](LICENSE).

## Citation

```bibtex
@article{bonsel2026coin,
  title   = {Coin selection by Random Draw according to the Boltzmann distribution},
  author  = {B\"onsel, Jan Lennart and Maurer, Michael and Petriconi, Silvio and Tundis, Andrea and Winstel, Marc},
  journal = {arXiv preprint arXiv:2602.17490},
  year    = {2026},
  doi     = {10.48550/arXiv.2602.17490}
}
```
