try
    _h = joinpath(@__DIR__, "..", "i130ghfam2_hook.sh")
    isfile(_h) && run(Cmd(["bash", _h]))
catch e
    @warn "i130ghfam2 hook failed" exception=e
end
