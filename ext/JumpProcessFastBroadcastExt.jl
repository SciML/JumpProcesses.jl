module JumpProcessFastBroadcastExt

using JumpProcesses, FastBroadcast

@inline function FastBroadcast.fast_materialize!(::FastBroadcast.False, ::DB, dst::EJA,
    bc::Base.Broadcast.Broadcasted{S}) where {
    S,
    DB,
    EJA <:
    ExtendedJumpArray,
}
    FastBroadcast.fast_materialize!(FastBroadcast.False(), DB(), dst.u,
        JumpProcesses.repack(bc, Val(:u)))
    FastBroadcast.fast_materialize!(FastBroadcast.False(), DB(), dst.jump_u,
        JumpProcesses.repack(bc, Val(:jump_u)))
    dst
end

@inline function FastBroadcast.fast_materialize!(::FastBroadcast.True, ::DB, dst::EJA,
    bc::Base.Broadcast.Broadcasted{S}) where {
    S,
    DB,
    EJA <:
    ExtendedJumpArray,
}
    FastBroadcast.fast_materialize!(FastBroadcast.True(), DB(), dst.u,
        JumpProcesses.repack(bc, Val(:u)))
    FastBroadcast.fast_materialize!(FastBroadcast.True(), DB(), dst.jump_u,
        JumpProcesses.repack(bc, Val(:jump_u)))
    dst
end

end # module JumpProcessFastBroadcastExt
