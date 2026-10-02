# Future real 2HDM input contract

No real samples or detector-level inference are run in this qualification task.
A future real study must first establish the exact parameters of interest, mass
points, coupling definition, width prescription and whether width varies with
coupling; native samples versus generator reweights; interference and signed
weight conventions; and cross-section, luminosity and rate metadata. These
must come from the actual generator setup, not a synthetic ratio toy.

For every proposed reweighted point calculate

    Neff(theta) = (sum_i abs(w_i))**2 / sum_i w_i**2
    C(theta) = abs(sum_i w_i) / sum_i abs(w_i)

Record the event population, sums and denominators. Invalid/zero denominators
must fail. These diagnostics assess Monte Carlo support and signed cancellation
before training or likelihood interpretation. A successful analytic toy does
not establish physical 2HDM sensitivity or general reconstruction benefit.
