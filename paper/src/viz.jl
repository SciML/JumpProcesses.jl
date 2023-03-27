half_page = (350, 175)
pgfkw = Dict(
    :size => half_page,
    :fontfamily => "times-serif",
    :titlefontsize => 7,
    :tickfontsize => 6,
    :labelfontsize => 7,
    :legendfontsize => 7,
)

@userplot QQPlot
@recipe function f(x::QQPlot)
    empirical_quant, expected_quant = x.args
    max_empirical_quant = maximum(maximum, empirical_quant)
    max_expected_quant = maximum(expected_quant)
    upperlim = ceil(maximum([max_empirical_quant, max_expected_quant]))
    @series begin
        seriestype := :line
        linecolor := :lightgray
        label --> ""
        (x) -> x
    end
    @series begin
        seriestype := :scatter
        aspect_ratio := :equal
        xlims := (0.0, upperlim)
        ylims := (0.0, upperlim)
        xaxis --> "Expected quantile"
        yaxis --> "Empirical quantile"
        markerstrokewidth --> 0
        markerstrokealpha --> 0
        markersize --> 1.5
        size --> (400, 500)
        label --> permutedims(["quantiles $i" for i = 1:length(empirical_quant)])
        expected_quant, empirical_quant
    end
end

@userplot BarcodePlot
@recipe function f(x::BarcodePlot)
    histories, = x.args
    k = length(histories)
    y = [fill(i, length(h)) for (i, h) in enumerate(histories)]
    seriestype := :scatter
    yticks --> 1:length(histories)
    alpha --> 0.5
    markerstrokewidth --> 0
    markerstrokealpha --> 0
    markershape --> :circle
    ylims --> (0.0, k + 1.0)
    for (i, h) in enumerate(histories)
        @series begin
            histories[i], y[i]
        end
    end
end
