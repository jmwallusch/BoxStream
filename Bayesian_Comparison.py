
# Six Nations 2025 — Bayesian Kick Analysis
# Compatible with the existing Python 3.7 environment.
# Keep this file in the same folder as 6N25_KICKS.xlsx.
# To run: python -m streamlit run 6N25_Bayesian.py

import os
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats


# ------------------------------------------------------------
# APP SETTINGS
# ------------------------------------------------------------

st.set_page_config(
    page_title="Six Nations 2025 — Box-Kick Analysis",
    layout="wide"
)

TEAM_COL = "Team"
OPP_COL = "Opp"
OUTCOME_COL = "Net Gain"

N_CHAINS = 4
N_ITER = 40000
BURN = 10000
THIN = 10

PARAMETER_NAMES = [
    "mu_group2",
    "log_sigma_group2",
    "mu_group_other",
    "log_sigma_group_other",
    "log(nu - 2)"
]


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

st.sidebar.header("Data")

uploaded_file = st.sidebar.file_uploader(
    "Optional: upload 6N25_KICKS.xlsx",
    type=["xlsx"]
)

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name="Data")
    except Exception as exc:
        st.error(
            "Could not read the uploaded Excel file: {}".format(exc)
        )
        st.stop()
else:
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "6N25_KICKS.xlsx"
    )

    if not os.path.exists(data_path):
        st.info(
            "Place 6N25_KICKS.xlsx in the same folder as this app, "
            "or upload it using the sidebar."
        )
        st.stop()

    try:
        df = pd.read_excel(data_path, sheet_name="Data")
    except Exception as exc:
        st.error(
            "Could not read 6N25_KICKS.xlsx, sheet 'Data': {}".format(exc)
        )
        st.stop()

required_cols = [TEAM_COL, OPP_COL, OUTCOME_COL]
missing_cols = [
    col for col in required_cols if col not in df.columns
]

if missing_cols:
    st.error(
        "Required columns missing from the data: {}".format(missing_cols)
    )
    st.stop()

df = df.copy()
df[OUTCOME_COL] = pd.to_numeric(
    df[OUTCOME_COL], errors="coerce"
)

teams = sorted(
    [str(value) for value in df[TEAM_COL].dropna().unique()]
)

if not teams:
    st.error("No team names were found in the Team column.")
    st.stop()

default_team_index = teams.index("IRE") if "IRE" in teams else 0

benchmark_team = st.sidebar.selectbox(
    "Kicking team",
    teams,
    index=default_team_index
)

team_rows = df.loc[
    df[TEAM_COL].astype(str) == benchmark_team
].copy()

opponents = sorted(
    [str(value) for value in team_rows[OPP_COL].dropna().unique()]
)

if not opponents:
    st.error(
        "No opponents were found for {}.".format(benchmark_team)
    )
    st.stop()

default_opp_index = opponents.index("WAL") if "WAL" in opponents else 0

selected_opponent = st.sidebar.selectbox(
    "Fielding opponent",
    opponents,
    index=default_opp_index
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "The comparison is always performed for the selected fielding opponent versus all other "
    "opponents faced by the kicking (benchmark) team."
)


# ------------------------------------------------------------
# PREPARE GROUPS
# ------------------------------------------------------------

team_rows = team_rows.loc[team_rows[OPP_COL].notna()].copy()

group_selected = team_rows.loc[
    team_rows[OPP_COL].astype(str) == selected_opponent,
    OUTCOME_COL
].dropna().to_numpy(dtype=float)

group_other = team_rows.loc[
    team_rows[OPP_COL].astype(str) != selected_opponent,
    OUTCOME_COL
].dropna().to_numpy(dtype=float)

n_selected = len(group_selected)
n_other = len(group_other)

if n_selected < 3 or n_other < 3:
    st.title(
        "Box-Kick Effectiveness: {} vs {}".format(
            benchmark_team, selected_opponent
        )
    )
    st.warning(
        "At least 3 valid Net Gain observations are required in each "
        "group. The current selection has {} and {}.".format(
            n_selected, n_other
        )
    )
    st.stop()

if (
    not np.all(np.isfinite(group_selected))
    or not np.all(np.isfinite(group_other))
):
    st.error("The selected data contain non-finite Net Gain values.")
    st.stop()


# ------------------------------------------------------------
# OBSERVED DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

# Observed medians, calculated directly from the data.
observed_median_selected = float(np.median(group_selected))
observed_median_other = float(np.median(group_other))

# Difference is selected opponent minus all other opponents.
observed_median_difference = (
    observed_median_selected - observed_median_other
)

# Sample standard deviations: ddof=1, using n - 1.
observed_sd_selected = float(
    np.std(group_selected, ddof=1)
)
observed_sd_other = float(
    np.std(group_other, ddof=1)
)

observed_sd_difference = (
    observed_sd_selected - observed_sd_other
)


# ------------------------------------------------------------
# BAYESIAN MODEL AND MCMC
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_bayesian_analysis(y_selected_tuple, y_other_tuple):
    """
    Student-t model and random-walk Metropolis sampler.

    Parameters:
    [mu_selected, log_sigma_selected,
     mu_other, log_sigma_other, log(nu - 2)].

    The groups have separate locations and scales but share nu > 2.
    """

    y_selected = np.asarray(y_selected_tuple, dtype=float)
    y_other = np.asarray(y_other_tuple, dtype=float)

    pooled = np.concatenate([y_selected, y_other])

    centre = np.median(pooled)
    outcome_scale = (
        1.4826 * np.median(np.abs(pooled - centre))
    )

    if not np.isfinite(outcome_scale) or outcome_scale <= 0:
        outcome_scale = np.std(pooled, ddof=1)

    if not np.isfinite(outcome_scale) or outcome_scale <= 0:
        raise ValueError(
            "Could not calculate a positive outcome scale."
        )

    z_selected = (y_selected - centre) / outcome_scale
    z_other = (y_other - centre) / outcome_scale

    def log_posterior(theta):
        (
            mu_selected,
            log_s_selected,
            mu_other,
            log_s_other,
            log_nu_minus_2
        ) = theta

        if not np.all(np.isfinite(theta)):
            return -np.inf

        if abs(log_s_selected) > 8 or abs(log_s_other) > 8:
            return -np.inf

        if log_nu_minus_2 < -8 or log_nu_minus_2 > 8:
            return -np.inf

        s_selected = np.exp(log_s_selected)
        s_other = np.exp(log_s_other)

        nu_minus_2 = np.exp(log_nu_minus_2)
        nu = 2.0 + nu_minus_2

        ll_selected = np.sum(
            stats.t.logpdf(
                z_selected,
                df=nu,
                loc=mu_selected,
                scale=s_selected
            )
        )

        ll_other = np.sum(
            stats.t.logpdf(
                z_other,
                df=nu,
                loc=mu_other,
                scale=s_other
            )
        )

        lp_mu = (
            stats.norm.logpdf(mu_selected, loc=0, scale=2)
            + stats.norm.logpdf(mu_other, loc=0, scale=2)
        )

        lp_scale = (
            stats.halfnorm.logpdf(s_selected, scale=1.5)
            + log_s_selected
            + stats.halfnorm.logpdf(s_other, scale=1.5)
            + log_s_other
        )

        lp_nu = (
            stats.expon.logpdf(nu_minus_2, scale=10)
            + log_nu_minus_2
        )

        result = ll_selected + ll_other + lp_mu + lp_scale + lp_nu

        if not np.isfinite(result):
            return -np.inf

        return result

    proposal_base = np.array([0.15, 0.08, 0.15, 0.08, 0.15])

    def run_chain(seed):
        rng = np.random.default_rng(seed)

        start = np.array([
            np.median(z_selected),
            np.log(max(np.std(z_selected, ddof=1), 0.2)),
            np.median(z_other),
            np.log(max(np.std(z_other, ddof=1), 0.2)),
            np.log(8.0)
        ])

        theta = start + rng.normal(0, proposal_base * 0.5)
        lp = log_posterior(theta)

        if not np.isfinite(lp):
            theta = start.copy()
            lp = log_posterior(theta)

        proposal_multiplier = 1.0
        accepted_total = 0
        accepted_window = 0
        saved = []

        for i in range(N_ITER):
            proposal = theta + rng.normal(
                0,
                proposal_base * proposal_multiplier
            )

            lp_proposal = log_posterior(proposal)

            if np.isfinite(lp_proposal):
                if np.log(rng.random()) < lp_proposal - lp:
                    theta = proposal
                    lp = lp_proposal
                    accepted_total += 1
                    accepted_window += 1

            # Adapt proposal size during burn-in only.
            if (i + 1) % 500 == 0 and (i + 1) <= BURN:
                rate = accepted_window / 500.0

                if rate < 0.15:
                    proposal_multiplier *= 0.85
                elif rate > 0.40:
                    proposal_multiplier *= 1.15

                proposal_multiplier = np.clip(
                    proposal_multiplier, 0.05, 10.0
                )

                accepted_window = 0

            if i >= BURN and (i - BURN) % THIN == 0:
                saved.append(theta.copy())

        return (
            np.asarray(saved),
            accepted_total / float(N_ITER)
        )

    chain_draws = []
    acceptance_rates = []

    for chain_number in range(N_CHAINS):
        draws, acceptance = run_chain(
            seed=2026 + chain_number
        )
        chain_draws.append(draws)
        acceptance_rates.append(acceptance)

    chains = np.stack(chain_draws, axis=0)

    # Posterior medians on the original Net Gain scale.
    selected_median_draws = (
        centre + outcome_scale * chains[:, :, 0]
    )

    other_median_draws = (
        centre + outcome_scale * chains[:, :, 2]
    )

    median_difference_draws = (
        selected_median_draws - other_median_draws
    ).ravel()

    median_results = {
        "selected_median": float(
            np.median(selected_median_draws)
        ),
        "other_median": float(
            np.median(other_median_draws)
        ),
        "difference_median": float(
            np.median(median_difference_draws)
        ),
        "difference_ci": np.quantile(
            median_difference_draws, [0.025, 0.975]
        ),
        "probability_selected_higher": float(
            np.mean(median_difference_draws > 0)
        )
    }

    # Observed sample SDs (ddof=1).
    sample_sd_selected = float(
        np.std(y_selected, ddof=1)
    )
    sample_sd_other = float(
        np.std(y_other, ddof=1)
    )

    sample_sd_difference = (
        sample_sd_selected - sample_sd_other
    )

    # Convert Student-t scale draws to population SDs.
    sigma_selected_draws = (
        outcome_scale * np.exp(chains[:, :, 1])
    )

    sigma_other_draws = (
        outcome_scale * np.exp(chains[:, :, 3])
    )

    nu_draws = 2.0 + np.exp(chains[:, :, 4])

    population_sd_selected_draws = (
        sigma_selected_draws
        * np.sqrt(nu_draws / (nu_draws - 2.0))
    )

    population_sd_other_draws = (
        sigma_other_draws
        * np.sqrt(nu_draws / (nu_draws - 2.0))
    )

    sd_difference_draws = (
        population_sd_selected_draws
        - population_sd_other_draws
    ).ravel()

    sd_results = {
        "sample_sd_selected": sample_sd_selected,
        "sample_sd_other": sample_sd_other,
        "sample_sd_difference": sample_sd_difference,
        "posterior_sd_selected": float(
            np.median(population_sd_selected_draws)
        ),
        "posterior_sd_other": float(
            np.median(population_sd_other_draws)
        ),
        "difference_median": float(
            np.median(sd_difference_draws)
        ),
        "difference_ci": np.quantile(
            sd_difference_draws, [0.025, 0.975]
        ),
        "probability_selected_higher": float(
            np.mean(sd_difference_draws > 0)
        )
    }

    # Basic R-hat diagnostic, matching the existing calculation.
    n_chains, n_draws, n_params = chains.shape

    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)

    within_variance = np.mean(chain_vars, axis=0)

    between_variance = (
        n_draws * np.var(chain_means, axis=0, ddof=1)
    )

    variance_hat = (
        ((n_draws - 1) / float(n_draws)) * within_variance
        + between_variance / n_draws
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(variance_hat / within_variance)

    return {
        "median": median_results,
        "sd": sd_results,
        "acceptance_rates": np.asarray(acceptance_rates),
        "rhat": np.asarray(rhat),
        "n_draws_per_chain": int(chains.shape[1]),
        "chains_shape": tuple(chains.shape)
    }


# ------------------------------------------------------------
# RUN BAYESIAN MODEL
# ------------------------------------------------------------

with st.spinner(
    "Running the Bayesian model (four MCMC chains). "
    "The first run for a particular comparison may take a little while..."
):
    try:
        results = run_bayesian_analysis(
            tuple(float(x) for x in group_selected),
            tuple(float(x) for x in group_other)
        )
    except Exception as exc:
        st.error(
            "The Bayesian analysis could not be completed: {}".format(exc)
        )
        st.stop()

median = results["median"]
sd = results["sd"]


# ------------------------------------------------------------
# 1. COMPARISON — OBSERVED DATA
# ------------------------------------------------------------

st.title(
    "Box-Kick Effectiveness: {} vs {}".format(
        benchmark_team, selected_opponent
    )
)

st.caption(
    "M6N 2025 data collected by J.M. Wallusch"
)

st.subheader("Comparison: median net metres gained")

# First row: observed medians.
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "{} vs {}".format(benchmark_team, selected_opponent),
        "{:.2f}".format(observed_median_selected)
    )
    st.caption("Observed median")

with col2:
    st.metric(
        "{} vs others".format(benchmark_team),
        "{:.2f}".format(observed_median_other)
    )
    st.caption("Observed median")

# Second row: difference in observed medians and Bayesian probability.
col3, col4 = st.columns(2)

with col3:
    st.metric(
        "Difference in observed medians",
        "{:+.2f}".format(observed_median_difference)
    )
    st.caption(
        "{} vs {} minus {} vs other opponents".format(
            benchmark_team,
            selected_opponent,
            benchmark_team
        )
    )

with col4:
    st.metric(
        "Probability of higher median gain",
        "{:.1f}%".format(
            100.0 * median["probability_selected_higher"]
        )
    )
    st.caption(
        "Bayesian posterior probability (see Technical Details below) that the median is higher "
        "against {}".format(selected_opponent)
    )

st.markdown("---")

# Third row: observed sample standard deviations.
st.subheader("Comparison: net metre variability")

sd_col1, sd_col2 = st.columns(2)

with sd_col1:
    st.metric(
        "{} vs {}".format(benchmark_team, selected_opponent),
        "{:.2f}".format(observed_sd_selected)
    )
    st.caption("Sample standard deviation")

with sd_col2:
    st.metric(
        "{} vs others".format(benchmark_team),
        "{:.2f}".format(observed_sd_other)
    )
    st.caption("Sample standard deviation")

# Fourth row: difference in observed SDs and Bayesian probability.
sd_col3, sd_col4 = st.columns(2)

with sd_col3:
    st.metric(
        "Difference in sample SDs",
        "{:+.2f}".format(observed_sd_difference)
    )
    st.caption(
        "{} vs {} minus {} vs other opponents".format(
            benchmark_team,
            selected_opponent,
            benchmark_team
    )
    )

with sd_col4:
    st.metric(
        "Probability of higher variability",
        "{:.1f}%".format(
            100.0 * sd["probability_selected_higher"]
        )
    )
    st.caption(
        "Bayesian posterior probability (see Technical Details below) that population SD is higher "
        "against {}".format(selected_opponent)
    )


# ------------------------------------------------------------
# 2. TECHNICAL DETAILS — POSTERIOR RESULTS
# ------------------------------------------------------------

st.markdown("---")
st.header("Technical Details")

st.subheader("Posterior estimates: median Net Gain")

st.caption(
    "These values are estimated from the posterior distributions, "
    "rather than calculated directly from the observed kicks."
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Posterior median: {} vs {}".format(
            benchmark_team, selected_opponent
        ),
        "{:.2f}".format(median["selected_median"])
    )

with col2:
    st.metric(
        "Posterior median: {} vs others".format(benchmark_team),
        "{:.2f}".format(median["other_median"])
    )

col3, col4 = st.columns(2)

with col3:
    st.metric(
        "Posterior median difference",
        "{:+.2f}".format(median["difference_median"])
    )

with col4:
    st.metric(
        "Probability selected-opponent median is higher",
        "{:.1f}%".format(
            100.0 * median["probability_selected_higher"]
        )
    )

median_ci = median["difference_ci"]

st.write(
    "**95% credible interval for the posterior median difference:** "
    "[{:.2f}, {:.2f}]".format(
        median_ci[0], median_ci[1]
    )
)

st.markdown("---")

st.subheader("Posterior estimates: population standard deviation")

st.caption(
    "The model's population SD estimates are distinct from the observed "
    "sample SDs shown in the Comparison section."
)

sd_col1, sd_col2 = st.columns(2)

with sd_col1:
    st.metric(
        "Posterior median SD: {} vs {}".format(
            benchmark_team, selected_opponent
        ),
        "{:.2f}".format(sd["posterior_sd_selected"])
    )

with sd_col2:
    st.metric(
        "Posterior median SD: {} vs others".format(benchmark_team),
        "{:.2f}".format(sd["posterior_sd_other"])
    )

sd_col3, sd_col4 = st.columns(2)

with sd_col3:
    st.metric(
        "Posterior difference in population SDs",
        "{:+.2f}".format(sd["difference_median"])
    )

with sd_col4:
    st.metric(
        "Probability selected-opponent SD is higher",
        "{:.1f}%".format(
            100.0 * sd["probability_selected_higher"]
        )
    )

sd_ci = sd["difference_ci"]

st.write(
    "**95% credible interval for the population SD difference:** "
    "[{:.2f}, {:.2f}]".format(
        sd_ci[0], sd_ci[1]
    )
)


# ------------------------------------------------------------
# MODEL DETAILS AND CONVERGENCE DIAGNOSTICS
# ------------------------------------------------------------

st.markdown("---")

with st.expander("Model details and convergence diagnostics"):

    st.write(
        "Model: two Student-t distributions with separate location "
        "and scale parameters and a shared degrees-of-freedom "
        "parameter constrained to be greater than 2."
    )

    st.write(
        "Sampling settings: {} chains × {:,} iterations; "
        "{:,} burn-in iterations; thinning interval {}. "
        "Retained draws per chain: {:,}.".format(
            N_CHAINS,
            N_ITER,
            BURN,
            THIN,
            results["n_draws_per_chain"]
        )
    )

    diagnostic_df = pd.DataFrame({
        "Parameter": PARAMETER_NAMES,
        "R-hat": results["rhat"]
    })

    st.dataframe(
        diagnostic_df,
        use_container_width=True
    )

    acceptance_df = pd.DataFrame({
        "Chain": [
            "Chain {}".format(i + 1)
            for i in range(N_CHAINS)
        ],
        "Acceptance rate": results["acceptance_rates"]
    })

    st.dataframe(
        acceptance_df,
        use_container_width=True
    )

    if (
        np.any(~np.isfinite(results["rhat"]))
        or np.any(results["rhat"] > 1.01)
    ):
        st.warning(
            "At least one R-hat is non-finite or exceeds 1.01. "
            "Investigate convergence before relying on the results."
        )
    else:
        st.write(
            "All R-hat values are at or below 1.01. This is encouraging, "
            "but R-hat alone does not prove convergence."
        )

st.caption(
    "Method note: the Student-t scale parameter is converted to "
    "population SD for every posterior draw using "
    "SD = scale × sqrt(nu / (nu − 2)). Differences are calculated "
    "draw by draw."
)
