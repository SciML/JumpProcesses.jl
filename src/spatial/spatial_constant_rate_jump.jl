struct SpatialConstantRateJump{F1, F2} <: AbstractJump
    """Function `rate(u,p,t,site)` that returns the jump's current rate."""
    rate::F1
    """Function `affect(integrator)` that updates the state for one occurrence of the jump."""
    affect!::F2
end
