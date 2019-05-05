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

function phiBtwn(state1::State, state2::State)
    angle = atan((state2.position[2] - state1.position[2]) / (state2.position[1] - state1.position[1]));
    correction = Int64(state2.position[1] - state1.position[1] < 0)*pi;
    angle + correction;
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

mutable struct BlueAgent <: Agent
    state::State
    detectRadius::Float64
    collisionRadius::Float64
    id::Int64
    threats::Array{RedAgent, 1}
    threatStateHats::Array{State, 1}
end

function BlueAgent(agentId::Int64, ybound::Float64 = 10.0)
    pos = [0.0, rand(Uniform(-ybound, ybound), 1)[1]];
    state = State(pos);
    BlueAgent(state, 5.0, 1.1, agentId, Array{RedAgent, 1}(), Array{State, 1}());
end

# Step the RedAgent forward.
# If it has reached [0,0], then return 1, o.w. 0
function truth!(r::RedAgent)
    step!(r);
    Int64(r.state.position[1] <= 0);
end

# returns the number of agents that have gotten to [0,0] this round
function truth!(redAgents::Array{RedAgent, 1})
    numPassed = 0;
    # update each RedAgent
    for i in 1:length(redAgents)
        trueIndex = i-numPassed;
        # check if it got to [0,0], if so, delete it
        if truth!(redAgents[trueIndex]) == 1
            deleteat!(redAgents, trueIndex);
            numPassed += 1;
        end
    end
    numPassed;
end   

# updates a state based on velocity
function updateState!(state::State, deltaT::Float64 = 1.0)
    state.position[1] += deltaT * state.velocity[1];
    state.position[2] += deltaT * state.velocity[2];
end

function checkCollision!(blue::BlueAgent, redAgents::Array{RedAgent, 1})
    for i in 1:length(redAgents)
        red = redAgents[i];
        # if the red agent is within the collision radius of the blue one
        # remove the red agent and return 1
        dist = distance(blue.state, red.state);
        if dist <= blue.collisionRadius
            deleteat!(redAgents, i);
            return(true);
        end
    end
    # if no red agents were in the collision radius, return 0
    false;
end

# moves the blue agents
function move!(blueAgents::Array{BlueAgent, 1}, redAgents::Array{RedAgent, 1})
    # numCollided contains the number of BlueAgents that have
    # collided with redAgents this round
    numCollided = 0;
    for i in 1:length(blueAgents)
        trueInd = i - numCollided;
        b = blueAgents[trueInd];
        # update the state of the BlueAGent
        updateState!(b.state);
        # if the blue and red agents are within blue's 
        # collision radius, annihilate the blue agent
        if checkCollision!(b, redAgents);
            numCollided += 1;
            deleteat!(blueAgents, trueInd);
        end
    end
end

# detect red agent within detection radius with probabiliy
# that drops off with distance from blue agent
function sense!(b::BlueAgent, r::RedAgent, sigma::Float64 = 1.0)
    b.threatPositions = [];
    dist = distance(b.state, r.state);
    if dist <= b.detectRadius
        if rand(Uniform(0.0, 1.0), 1)[1] < 1 - dist/b.detectRadius
            append!(b.threatPositions, r.state.position);
        end
    end
end

function sense!(blueAgents::Array{BlueAgent, 1}, redAgents::Array{RedAgent, 1})
    for b in blueAgents
        for r in redAgents
            sense!(b, r);
        end
    end
end

function communicate!(b1::BlueAgent, b2::BlueAgent)
    if distance(b1.state, b2.state) < b1.detectRadius
        union!(b1.threatPositions, b2.threatPositions);
        union!(b2.threatPositions, b1.threatPositions);
    end
end

function communicate!(blueAgents::Array{BlueAgent, 1})
    for i in 1:length(blueAgents)-1
        b1 = blueAgents[i];
        for j in i+1:length(blueAgents)
            b2 = blueAgents[j];
            communicate!(b1, b2);
        end
    end
end

function ekf!(z::Array{Float64, 1}, oldYhat::Array{Float64, 1}, oldP::Array{Float64, 2}, deltaT::Float64 = 1.0)
    capPhi = transpose(reshape([1, 0, deltaT, 0, 0, 1, 0, deltaT, 0, 0, 1, 0, 0, 0, 0, 1], (4,4)));
    Q = reshape([0.3, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0.05, 0, 0, 0, 0, 0.05], (4,4));
    # predict
    yhatPredict = capPhi * oldYhat;
    pPredict = capPhi * oldP * transpose(capPhi) + Q;
    
    # correct
    eta = z
end

function infer!(b::BlueAgent)
    for z in b.threatRelativeStateHats
        (yhat, P) = ekf!(z, oldYhat, oldP);
    end
end

function getRelativeStates!(b::BlueAgent, sigmaPhi::Float64 = 0.05, sigmaRho::Float64 = 0.1)
    b.threatRelativeStateHats = [];
    R = reshape([sigmaPhi, 0.0, 0.0, sigmaRho], (2,2));
    for r in b.threats
        phi = phiBtwn(b.state, r.state);
        dist = distance(b.state, r.state);
        append!(b.threatRelativeStateHats, [phi, dist] + rand(MvNormal([0.0, 0.0], R)));
    end
    infer!(b);
end

function infer!(blueAgents::Array{BlueAgent, 1})
    for b in blueAgents
        infer!(b);
    end
end

# find the closest red agent if there is one
function findClosestRed(b::BlueAgent)
    minDist = Inf;
    closestState = nothing;
    for state in b.threatStateHats
        dist = distance(b.state, state);
        if dist < minDist
            minDist = dist;
            closestState = state;
        end
    end
    state;
end

# update velocity to move in direction of closest red agent
# if b sees no red agents, then continue on current trajectory
function decide!(b::BlueAgent, gain::Float64 = 0.8, deltaT::Float64 = 1.0)
    r = findClosestRed(b);
    if r != nothing
        # calculate gradient
        gradHat = b.state.position + deltaT*b.state.velocity - r.state.position;
        # update velocity
        b.state.velocity -= gain/deltaT^2*gradHat;
    end
end

# calculate velocity update by minimizing loss
function decide!(blueAgents::Array{BlueAgent, 1})
   for b in blueAgents
        decide!(b);
    end
end

function main(n::Int64 = 1)
    # Create redAgents
    redAgents = [RedAgent() for i in 1:n];
    blueAgents = [BlueAgent(i, Array{Float64, 1}()) for i in 1:n];
    numPassed = 0;
    
    while !isempty(redAgents)
        numPassed += truth!(redAgents);
        decide!(blueAgents);
        move!(blueAgents, redAgents);
        sense!(blueAgents, redAgents);
        communicate!(blueAgents);
        infer!(blueAgents);
    end
    
    println(string(n-numPassed, " of ", n, " were successfully intercepted."));
end