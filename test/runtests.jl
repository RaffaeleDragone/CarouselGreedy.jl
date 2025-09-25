using Test
using CarouselGreedy

@testset "CarouselGreedy basic tests" begin
    solver = CarouselGreedySolver((s,sol)->length(sol) >= 2, (s,sol,c)->rand())
    solution = minimize(solver)
    @test length(solution) >= 0   # test banale per vedere se gira
end