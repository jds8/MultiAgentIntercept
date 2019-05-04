import Pkg;
Pkg.add("Polynomials");
using Polynomials;

Pkg.add("Random");
using Random;

Pkg.add("Distributions");
using Distributions;

Pkg.add("NLsolve");
using NLsolve;

Pkg.add("PyPlot");
using PyPlot;


Random.seed!(seed)

seed = 1234
Random.seed!(seed);
# Create a polynomial from [0,0] to [x,y]
function getTrajFromStartX(x, y, ybound::Float64=10.0)
    # generate points of polynomial
    deg = rand(2:4, 1)[1];
    partitionLength = x / deg;
    xvals = partitionLength*collect(0:deg);
    unif = Uniform(-ybound,ybound);
    yvals = append!([0.0], rand(unif, deg-1));
    append!(yvals, y);
    polyfit(xvals,yvals);
end


mutable struct State
    position::Array{Float64, 1}
    velocity::Array{Float64, 1}
    function State(pos::Array{Float64, 1})
        # Set velocity to be 1 unit in the East direction
        vel = [1.0, 0.0];
        new(pos, vel);
    end
end

function distance(state1::State, state2::State)
    state1.position - state2.position |> x->x.^2 |> sum |> sqrt
end

abstract type Agent end

mutable struct RedAgent <: Agent
    state::State
    trajectory::Poly
end

# Red Agent inner constructors
# RedAgents should be placed at a north position randomly between -10 and 10 units
function RedAgent(ybound::Float64 = 10.0) 
    pos = [10.0, rand(Uniform(-ybound, ybound), 1)[1]];
    state = State(pos);
    traj = getTrajFromStartX(pos[1], pos[2]);
    RedAgent(state, traj);
end
RedAgent(poly::Poly) = RedAgent(State([10.0, rand(Uniform(-10.0, 10.0), 1)[1]]), poly);

function step!(red::RedAgent)
    oldx = red.state.position[1];
    oldy = red.state.position[2];
    slope = polyder(red.trajectory)(oldx);
    xdelta = -abs(sin(atan(-1/slope)));
    newx = oldx + xdelta;
    newy = red.trajectory(newx);
    red.state.position[1] = newx;
    red.state.position[2] = newy;
end

# Step the RedAgent forward.
# If it has reached [0,0], then return 1, o.w. 0
function truth!(red::RedAgent)
    step!(r);
    Int64(r.state.position[1] <= 0);
end

# returns the number of agents that have gotten to [0,0] this round
function truth!(redAgents::Array{RedAgent, 1})
    numPassed = 0;
    # update each RedAgent
    for i in 1:length(redAgents)
        trueIndex = i-numPassed
        # check if it got to [0,0], if so, delete it
        if truth!(redAgents[trueIndex]) == 1
            deleteat!(redAgents, trueIndex);
            numPassed += 1;
        end
    end
    numPassed;
end   
mutable struct BlueAgent <: Agent
    state::State
    detectRadius::Float64
    collisionRadius::Float64
    id::Int64
    threats::Array{RedAgent, 1}
end

function BlueAgent(agentId::Int64, threats::Array{RedAgent, 1}, ybound::Float64 = 10.0)
    pos = [0.0, rand(Uniform(-ybound, ybound), 1)[1]];
    state = State(pos);
    id = agentId;
    BlueAgent(state, 3.0, 0.1, id, threats);
end

function beginDetect(blue::BlueAgent)
    blue.threats = [];
end

function beginDetect(blueAgents::Array{BlueAgent, 1})
    for blue in blueAgents
        beginDetect(blue);
    end
end

function detect!(blue::BlueAgent, ally::BlueAgent)
    if distance(blue.state, ally.state) < blue.radius
        
    end
end

function detect!(blue::BlueAgent, threat::RedAgent)
    dist = distance(blue.state, threat.state);
    if dist <= blue.radius
        if rand(Uniform(0.0, 1.0), 1) < 1 - dist/blue.radius
            append!(threats, threat)
        end
    end
end


# moves a blue agent according to its velocity
function move!(blue::BlueAgent)
end

# moves the blue agents
function move!(blueAgents::Array{BlueAgent, 1})
    for i in 1:length(blueAgents)
        move!(blueAgents[i])
    end
end

function main(n::Int64 = 5)
    # Create redAgents
    redAgents = [RedAgent() for i in 1:n];
    numPassed = 0;
    
    while !isempty(redAgents)
        numPassed += truth!(redAgents);
        #decide()
        move()
        #sense()
        #communicate()
        #infer()
    end
    
    println(string(n-numPassed, " of ", n, " were successfully intercepted."));
end

main()