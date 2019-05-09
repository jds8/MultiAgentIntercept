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

Pkg.add("LinearAlgebra");
using LinearAlgebra;

seed = 1234
Random.seed!(seed);
# Create a polynomial from [0,0] to [x,y]
function getTrajFromStartX(x::Float64, y::Float64, ybound::Float64=10.0)
    # generate points of polynomial
    deg = rand(2:4, 1)[1];
    partitionLength = x / deg;
    xvals = partitionLength*collect(0:deg);
    unif = Uniform(-ybound,ybound);
    yvals = append!([0.0], rand(unif, deg-1));
    push!(yvals, y);
    polyfit(xvals,yvals);
end

Random.seed!(seed)

mutable struct State
    position::Array{Float64, 1}
    velocity::Array{Float64, 1}
    function State(pos::Array{Float64, 1}, vel::Array{Float64, 1})
        new(pos, vel);
    end
end

function State(pos::Array{Float64, 1})
    # Set velocity to be 1 unit in the East direction
    vel = [1.0, 0.0];
    State(pos, vel);
end

function asVec(state::State)
    vcat(state.position, state.velocity);
end

function distance(x::Array{Float64, 1}, y::Array{Float64, 1})
    x - y |> x->x.^2 |> sum |> sqrt
end

function distance(state1::State, state2::State)
    distance(state1.position, state2.position);
end

function distance(state1::State, y::Array{Float64, 1})
    distance(state1.position, y);
end

function phiBtwn(x::Array{Float64, 1}, y::Array{Float64, 1})
    angle = atan((y[2] - x[2]) / (y[1] - x[1]));
    correction = Int64(y[1] - x[1] < 0)*pi;
    angle + correction;
end

function phiBtwn(state1::State, state2::State)
    phiBtwn(state1.position, state2.position);
end

function phiBtwn(state1::State, y::Array{Float64, 1})
    phiBtwn(state1.position, y);
end

function getStateFromRelative!(state::State, relOffset::Array{Float64, 1})
    state.position[1] += relOffset[2]*cos(relOffset[1]);
    state.position[2] += relOffset[2]*sin(relOffset[1]);
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
    threatStates::Array{State, 1}
    threatRelativeStateHat::Array{Float64, 1}
    threatStateHat::State
    oldP::Array{Float64, 2}
end

function BlueAgent(agentId::Int64, ybound::Float64 = 10.0)
    pos = [0.0, rand(Uniform(-ybound, ybound), 1)[1]];
    state = State(pos);
    nullP = zeros(4,4);
    # the second to last argument is state arbitrarily
    # threatStateHat will not be used until oldP != zeros(4,4)
    BlueAgent(state, 5.0, 1.1, agentId, Array{State, 1}(), 
              Array{Float64, 1}(), state, nullP);
end

BlueAgent(1)

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
    b.threatStates = [];
    dist = distance(b.state, r.state);
    if dist <= b.detectRadius
        if rand(Uniform(0.0, 1.0), 1)[1] < 1 - dist/b.detectRadius
            push!(b.threatStates, r.state);
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
        union!(b1.threatStates, b2.threatStates);
        union!(b2.threatStates, b1.threatStates);
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

function getPredictions(b::BlueAgent, deltaT::Float64 = 1.0)
    capPhi = transpose(reshape([1, 0, deltaT, 0, 0, 1, 0, deltaT, 0, 0, 1, 0, 0, 0, 0, 1], (4,4)));
    Q = reshape([0.3, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0.05, 0, 0, 0, 0, 0.05], (4,4));
    # predict
    yhatPred = capPhi * asVec(b.threatStateHat);
    pPred = capPhi * b.oldP * transpose(capPhi) + Q;
    
    # correct
    phiCorrect = phiBtwn(b.state, yhatPred);
    distCorrect = distance(b.state, [yhatPred[1], yhatPred[2]]);
    z = b.threatRelativeStateHat;
    eta = z - [phiCorrect, distCorrect];
    
    (yhatPred, pPred, eta, distCorrect);
end
    
function ekf!(b::BlueAgent, R::Array{Float64, 2}, deltaT::Float64 = 1.0)
    (yhatPred, pPred, eta, distCorrect) = getPredictions(b, deltaT);
    
    eastDiff = b.state.position[1] - yhatPred[1];
    northDiff = b.state.position[2] - yhatPred[2];
    H = reshape([(northDiff)/(distCorrect^2), -(eastDiff)/(distCorrect), 
            -(eastDiff)/(distCorrect^2), -(northDiff)/(distCorrect), 0, 0, 0, 0], (2,4));
    S = H*pPred*transpose(H) + R;
    
    K = pPred*transpose(H)*inv(S);
    yhat = yhatPred + K*eta;
    P = (Array{Float64, 2}(I, 4, 4) - K*H)*pPred;
    
    # store estimates
    b.threatStateHat = State([yhat[1], yhat[2]], [yhat[3], yhat[4]]);
    b.oldP = P;
end

function setNewThreatStateHat!(b::BlueAgent)
    getStateFromRelative!(b.threatStateHat, b.threatRelativeStateHat);
    b.threatStateHat.velocity[1] = -b.state.velocity[1];
    b.threatStateHat.velocity[2] = -b.state.velocity[2];
    updateState!(b.threatStateHat);
end

function infer!(b::BlueAgent, R::Array{Float64, 2}, etaThreshold::Float64 = 2.0)
    # if we have a threatRelativeStateHat, then do an ekf update
    if length(b.threatRelativeStateHat) > 0
        # if the covariance is all zeros, then this is a new
        # red agent target
        if b.oldP != zeros(4,4)
            # suppose that the red agent was moving with the opposite velocity as b
            setNewThreatStateHat!(b);
        else
            (_, _, eta, _) = getPredictions(b);
            # if the new state is too far from the old state, then set red agent
            # as new threat
            if norm(eta, 2) > etaThreshold
                setNewThreatStateHat!(b);
            end
        end
        ekf!(b, R);
    end
end

function getRelativeStates!(b::BlueAgent, R::Array{Float64, 2})
    b.threatRelativeStateHat = Array{Float64, 1}();
    # if b is not aware of any threats, set covariance
    # back to zeros and return
    if length(b.threatStates) == 0
        b.oldP = zeros(4,4);
        return
    end
    # get all relative red agent states
    relStates = Array{Array{Float64, 1}, 1}();
    for rstate in b.threatStates
        phi = phiBtwn(b.state, rstate);
        dist = distance(b.state, rstate);
        push!(relStates, [phi, dist] + rand(MvNormal([0.0, 0.0], R)));
    end
    # find closest red agent to b
    minDist = Inf;
    for relState in relStates
        if relState[2] < minDist
            minDist = relState[2];
            b.threatRelativeStateHat = relState;
        end
    end
end

function infer!(blueAgents::Array{BlueAgent, 1}, sigmaPhi::Float64 = 0.05, sigmaRho::Float64 = 0.1)
    R = reshape([sigmaPhi, 0.0, 0.0, sigmaRho], (2,2));
    for b in blueAgents
        getRelativeStates!(b, R);
        infer!(b, R);
    end
end

# update velocity to move in direction of closest red agent
# if b sees no red agents, then continue on current trajectory
function decide!(b::BlueAgent, gain::Float64 = 0.8, deltaT::Float64 = 1.0)
    rstate = b.threatStateHat
    if rstate != nothing
        # calculate gradient
        gradHat = b.state.position + deltaT*b.state.velocity - rstate.position;
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
    blueAgents = [BlueAgent(i) for i in 1:n];
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