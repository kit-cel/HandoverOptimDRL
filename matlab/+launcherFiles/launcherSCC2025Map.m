% LAUNCHER FILE FOR THE SCC 2025
%   - Map generation
% see also scenarios.scenarioSCC2025Map

close all;
clear;
clc;

simulationID = string(datetime('now','Format','yyyyMMdd_HHmmss'));

% Parameters
latitude                = sort([49.004900, 49.011200]); 
longitude               = sort([8.377300,   8.395600]);
stepSize                = 10;
minNumSamplesPerCell    = 1;

% Launch simulations
simulationMode = parameters.setting.SimulationType.local;
result = simulate(@(params)scenarios.scenarioSCC2025Map(params, latitude, longitude, stepSize), simulationMode);