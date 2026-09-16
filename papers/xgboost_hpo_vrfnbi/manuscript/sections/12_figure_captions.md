## Figure captions

**Figure 1. The arm lattice.** Each arrow is a single-factor contrast and the label
names the one thing it varies; everything else is held fixed by a run-time fingerprint
over the solver configuration, realizer, weight grid, surrogate identities and
reference. The two primary contrasts are marked.

**Figure 2. The geometry contrast, per replication.** Paired differences in
CORE-relative hypervolume ratio, NBI-S minus WS-S, sorted within each dataset, $R = 30$,
CORE reference. Blue favours NBI-S. The dashed line is the median. Note that the
vertical scales differ by dataset.

**Figure 3. Dispersion across the panel.** The same paired differences as Figure 2,
shown as distributions. The boundary dataset's spread is roughly thirty times MAGIC's,
which is why its contrast does not resolve at $R = 30$ despite a larger point estimate.

**Figure 4. Anchor provenance and the associated geometry.** Left: the full
NBI-S $\rightarrow$ NBI-R difference beside the anchor-injection control, which inserts
NBI-R's empirical anchors into NBI-S's set and changes nothing else. Right: the CHIM
extent under empirical anchors relative to surrogate-derived anchors, as a per-dataset
median. The juxtaposition is an association, not a demonstrated cause, and it does not
hold on MAGIC.

**Figure 5. Reference sensitivity.** Median paired difference under the CORE reference
(primary, horizontal) against the AUGMENTED reference (mandatory sensitivity,
vertical), for all twelve dataset-by-contrast cells. Red marks the single cell whose
significance differs between references; no cell changes direction.

**Figure 6. Surrogate-gate regime and baselines.** Left: per-dataset pass rates of the
frozen external reliability criterion over $R = 30$; the gate is diagnostic and filters
nothing. Right: median CORE-relative hypervolume ratio by method. The dashed rule marks
the CORE reference itself and is **not** an upper bound — a ratio above it means the
candidate set improved on the finite reference.
