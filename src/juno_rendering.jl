@require Juno begin
  # Juno Rendering
  Juno.render(i::Juno.Inline, x::ExtendedJumpArray) = Juno.render(i, Juno.defaultrepr(x))
  Juno.render(i::Juno.Inline, x::CoupledArray) = Juno.render(i, Juno.defaultrepr(x))
  Juno.render(i::Juno.Inline, x::JumpProblem) = Juno.render(i, Juno.defaultrepr(x))
end
