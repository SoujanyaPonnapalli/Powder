# AZ-aware placement and fault sensitivity

Status: version-two library profile and engine support, 2026-10-28.

The existing `region_outage/aws-us-from-azure` and historical catalogue remain
unchanged. Their all-region WAN process is a *stress* interpretation of two
Azure connectivity incidents, not a demonstrated complete loss of every AWS
US AZ. The new scenario is `scenario/canonical-central-az-aware@v2`, citing
`region_outage/aws-us-az-aware@v2` and requiring the catalogue's
`az_aware=True` version. The primary model retains per-region outages but does
not inject the old all-region WAN process; its rate and source IDs remain in
`wan_stress` metadata for a separate sensitivity. Partial WAN degradation is
**not yet a parameterised correlated event process** and must not be described
as one in results.

## Binding and scopes

- Catalogue machines have four AWS US regions and three logical AZ IDs per
  region (`<region>:az1..3`). A seeded shuffle assigns machines within each
  region almost equally among the three AZs. These logical IDs are stable
  across runs, not account-specific AWS AZ letters. The old catalogue file is
  immutable; `build_catalogue.py --az-aware` writes a new `-az-v2` file.
- The placement candidate generator drops a candidate **before Markov/MC** if
  any region holds more than half of its replicas, or (for AZ-aware catalogues)
  two replicas share an AZ. RF 3 therefore spans three regions; RF 5 and RF 7
  may use at least three regions, provided no one region holds a quorum.
- An independent AZ outage affects only the chosen AZ. A truly regional event
  still affects all three AZs in that region. The historical all-four-region
  process remains available only with the old scenario/stress model.
- Both Python and Rust `NodeConfig` and `NetworkConfig` accept optional AZ IDs
  and per-AZ processes. Old jobs without AZ fields retain their behavior.

## AZ rate proxy and uncertainty

Two verified Azure PIR measurements explicitly report one-AZ compute outages:
`azure-pir-incidents#2022-06-07-eus2-cooling` (709 min) and
`azure-pir-incidents#2023-09-16-eus-power` (696 min). The primary model pools
them over 4 mapped AWS regions × 3 logical AZs × the 2022-06-01 to
2026-10-20 observation window. A Jeffreys Gamma-Poisson posterior yields
**0.0475 events per AZ-year** central; the plotted duration is the two-event
mean **702.5 min**, with observed endpoints as low/high sensitivities. These
numbers are a *cross-cloud proxy*, not directly measured AWS AZ rates. Public
PIR coverage and the rate tail are uncertain; West US's source region does not
even expose AZs. The 2025 East US 2 networking event was excluded because its
record describes cross-zone impact despite its CSV classification.

The old `region_wide` / `multi_az` counts are conservatively left in the
per-region process. A two-AZ incident can still be over-modeled as all three
AZs unavailable. A later correlated-failure revision should split regional
scope and assign partial WAN impact explicitly using new versioned source and
profile entries; do not silently overwrite this `@v2` scenario after results
cite it.

## Pilot feasibility and comparison limits

One 100,000-run AZ-aware Rust MC probe per RF, using a region-balanced
cheapest-first group from a seeded catalogue, gave representative mean
availability of roughly **4.35, 4.55, and 4.82 nines** at RF 3/5/7. At the
current five-nines target all three were below the threshold; the user chose
**4.5 nines** for the comparative hybrid study. This is not a proof that no
five-nines placement exists. The 10,000-RSM ILP is free to choose RF 3/5/7:
no mix is imposed, and selected proportions must be reported as observed.
Availability plots will state whether each mark is Markov-screened, MC-scored,
or MC-certified. Mean per-RSM availability is not whole-deployment uptime when
hundreds of RSMs share a machine.

No additional VM is needed. Rebuild the new source-derived entries with
`python3 scripts/fault_library/build_az_outages.py`; use
`python3 scripts/fault_library/build_catalogue.py --az-aware` for the new
immutable catalogue path. The independent full library build also calls the
AZ builder after the historical region builder.
