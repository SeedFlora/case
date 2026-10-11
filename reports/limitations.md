# Limitations

## Sample and protocol

- **30 participants.** With participant-level aggregation the effective sample size for every statistical test is n = 30, not the 6,810 windows. Confidence intervals are correspondingly wide, and modest effects cannot be resolved.
- **Laboratory video-elicitation protocol.** Participants sat and watched short clips. Nothing here demonstrates transfer to ambulatory, occupational, or clinical settings.
- **Shared stimulus set.** All participants viewed the same 8 clips. The protocol is subject-independent but *not* stimulus-independent, so it does not evidence generalisation to unseen stimuli.
- **Continuous affect is not happiness and not a clinical state.** The targets are continuous self-reported joystick positions on a valence-arousal plane. No medical or diagnostic claim is made or supported.

## Measurement and preprocessing

- **Annotation is a motor act with unknown lag.** A 0-3 s delay sweep on validation participants produced no stable improvement, so no shift is applied; a participant-specific lag may nevertheless exist and would be absorbed as noise.
- **Preprocessing choices are consequential.** Modality-specific filters, the 64 Hz analysis grid, and the sEMG RMS-envelope decision all shape the inputs. Window length was selected on validation participants (20 s beat 10 s and 5 s consistently), but the broader preprocessing design was fixed a priori and not searched.
- **One participant (21) shows ~20 C peripheral skin temperature** throughout. It is retained as physiologically plausible vasoconstriction, but it is an outlier in the covariate distribution.

## Robustness experiments

- **Corruption is synthetic.** Gaussian noise, flatlines, sample stalls, clipping and impulses are stand-ins for sensor faults. They do not reproduce the full behaviour of a detaching electrode, motion artifact, or a failing amplifier, and results should not be read as a hardware reliability guarantee.
- **Missing sensors are simulated by masking**, i.e. a channel is absent for the whole session. Intermittent dropout, partial degradation, and correlated multi-sensor failures are not covered.
- **Corruption affects half the channels by design.** Corrupting all channels equally would leave a reliability gate nothing to discriminate; the chosen protocol is therefore favourable to gating in principle, and this is stated rather than hidden.

## Modelling and comparison

- **No comparison to published CASE numbers.** Prior CASE work uses the EPiC 2023 challenge partitions, which are not the participant-grouped protocol used here. Cross-paper numbers are therefore not comparable and no such comparison is claimed.
- **Compact models only.** Encoders are deliberately small (~0.2 M parameters) because of the participant count. Larger architectures were not searched and might behave differently.
- **Limited hyper-parameter search.** A small predeclared set was tuned on validation participants and frozen. No architecture search was performed.
- **Architectural novelty is not claimed.** Quality-weighted fusion and modality dropout are both established in prior work; the contribution is the leakage-safe subject-independent robustness benchmark on CASE.
