function [params] = karlsruheScenario(params, index_output)
    % input:
    %   params: [1x1]handleObject parameters.Parameters
    %
    % output:
    %   params: [1x1]handleObject parameters.Parameters
    %
    % initial author: Johannes Voigt
    %% General Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    params.postprocessor = simulation.postprocessing.MediumPP;
    %% time: n_chunks * slots_per_chunk = simulation steps
    % time config
    params.time.numberOfChunks              = 100;   % a sufficently large number of chunks to achieve paralleization gain
    params.time.slotsPerChunk               = 15;	% the first 3 slots in a chunk are discarded, since no feedback is available
    params.time.timeBetweenChunksInSlots    = 0;	% the chunks should be independent
    params.time.slotDuration                = 0.12;
    params.useFeedback = false;

    % set the carrier frequency and bandwidth
    params.carrierDL.centerFrequencyGHz             = 1.8;    % GHz
    params.transmissionParameters.DL.bandwidthHz    = 5e6;  % Hz
    
    % disable HARQ - is not implemented for a feedback delay larger than 1
    params.useHARQ = false;
    
    % define the region of interest
    params.regionOfInterest.xSpan = 1400;
    params.regionOfInterest.ySpan = 800;
    params.regionOfInterest.zSpan = 50;

    %% Additional Results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    params.save.losMap              = true;
    params.save.isIndoor            = true;
    params.save.antennaBsMapper     = true;
    params.save.macroscopicFading   = true;
    params.save.wallLoss            = true;
    params.save.shadowFading        = true;
    params.save.antennaGain         = true;
    params.save.receivePower        = true;
    params.save.pathlossTable       = true;
    %params.save.feedback            = true;
    %params.save.userScheduling      = true;

    
    %% Shadow Fading %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    params.shadowFading.on          = true;
    params.shadowFading.resolution	= 5;            % default value
    params.shadowFading.mapCorr     = 0.5;          % default value
    params.shadowFading.meanSFV     = 0;            % default value
    params.shadowFading.stdDevSFV	= 1;            % default value
    params.shadowFading.decorrDist	= 20*log(2);    % default value

    
    %% Path Loss Models %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % set path loss model for each link type
    indoor	= parameters.setting.Indoor.indoor;
    outdoor	= parameters.setting.Indoor.outdoor;
    LOS     = parameters.setting.Los.LOS;
    NLOS	= parameters.setting.Los.NLOS;
    
    % set path loss models for macro base station
    % set path loss models
    macro = parameters.setting.BaseStationType.macro;

    params.pathlossModelContainer.modelMap{macro,	indoor,     LOS}    = parameters.pathlossParameters.UrbanMacro5G;
    params.pathlossModelContainer.modelMap{macro,	indoor,     NLOS}   = parameters.pathlossParameters.UrbanMacro5G;
    params.pathlossModelContainer.modelMap{macro,	outdoor,    LOS}    = parameters.pathlossParameters.UrbanMacro5G;
    params.pathlossModelContainer.modelMap{macro,	outdoor,	NLOS}   = parameters.pathlossParameters.UrbanMacro5G;
    params.pathlossModelContainer.modelMap{macro,	outdoor,	LOS}.isLos = true;
    params.pathlossModelContainer.modelMap{macro,	outdoor,	NLOS}.isLos = false;

    %% Blockage / Buildings & Walls %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    openStreetMapCity = parameters.city.OpenStreetMap;
    openStreetMapCity.latitude          = [49.0049, 49.0112];
    openStreetMapCity.longitude         = [8.3773,  8.3956];   
    openStreetMapCity.streetWidth       = 10;
    openStreetMapCity.estBuildingHeight = true;                   % estimate the building heights based on the number of floors
    openStreetMapCity.wallLossdB        = 3;
    params.cityParameters('OSMCity') = openStreetMapCity;

    % Base station coordinates
    bsLatitude  = [49.01089, 49.01009, 49.01024, 49.0053, 49.00534];   %[49.01109, 49.00531, 49.01076, 49.01004, 49.00528];
    bsLongitude = [8.37934, 8.39415, 8.38630, 8.3830, 8.39026];   %[8.37935, 8.38556, 8.38703, 8.39501, 8.39483];
    bsHeights   = [0, 0, 0, 0, 0];
    bsPositions = tools.gcsToMeter(bsLatitude, bsLongitude, bsHeights, openStreetMapCity);
    azimuthList = [250, 290, 270, 90, 90];   %[330, 180, 280, 180, 100];
    elevationList = [100, 100, 130, 100, 90];
    %% Base Stations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for i = 1:5
        antenna = parameters.basestation.antennas.ThreeSector;
        antenna.nTX          = 4;
        antenna.azimuth      = azimuthList(i);
        antenna.elevation    = elevationList(i);
        antenna.technology   = parameters.setting.NetworkElementTechnology.NRMN_5G;
        antenna.precoderAnalogType	= parameters.setting.PrecoderAnalogType.none;

        % base station
        baseStation                     = parameters.basestation.PredefinedPositions;
        baseStation.antenna.baseStationType    = parameters.setting.BaseStationType.macro;
        baseStation.positions	        = [bsPositions(1,i); bsPositions(2,i)];
        baseStation.nSectors            = 1;
        baseStation.antenna	            = antenna;
        baseStation.antenna.height= 30;               % antenna height
        params.baseStationParameters(strcat('BaseStation',num2str(i))) = baseStation;
    end

    %% Users %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % predefined position for each slot
    % Input: edge points
    % Output: coordinates for each slot between these
    % edge points depending on speed and slot_time
    % author: Peter Gu
    ueheight = 1.5;
    predefinedUser = parameters.user.PredefinedPositions;
    predefinedUser.speed                        = 50/3.6; % this speed will only affect the Doppler shift in the channel model

    speed = predefinedUser.speed;
    % Enter edge points here

    T = readtable(strcat('routes/test/output', char(index_output), '.csv'));
    T_x = T.Var1;
    T_y = T.Var2;
    edge_points = zeros(size(T_x, 1),2);
    for i_T = 1:size(T_x, 1)
        edge_points(i_T, 1) = T_x(i_T);
        edge_points(i_T, 2) = T_y(i_T);
    end
    edge_heights = zeros(size(edge_points, 1),1);
    edge_points_calc = tools.gcsToMeter(edge_points(:,1), edge_points(:,2), ueheight, openStreetMapCity);
    edge_points_ = edge_points_calc(1:2,:);
    positionList = zeros(2, params.time.nSlotsTotal);
    i_slot = 1;
    i_edge_pos = 1;
    while (i_edge_pos <= size(edge_points_, 2)-1) && (i_slot <= params.time.nSlotsTotal-1)
        % Starting point
        positionList(:,i_slot) = edge_points_(:,i_edge_pos);
        % Start of calculation of direction
        dir_x = edge_points_(1, i_edge_pos+1) - edge_points_(1, i_edge_pos);
        dir_y = edge_points_(2, i_edge_pos+1) - edge_points_(2, i_edge_pos);
        % Direction calculated using factor and pythagoras
        factor = (speed*params.time.slotDuration)^2/(dir_x^2+dir_y^2);
        Doffset = [sqrt(factor)*dir_x; sqrt(factor)*dir_y];
        % Stopping condition to move to next edge point
        if edge_points_(1, i_edge_pos) < edge_points_(1, i_edge_pos+1)
            while positionList(1, i_slot) < edge_points_(1, i_edge_pos+1) && (i_slot <= params.time.nSlotsTotal-1)
                positionList(:,i_slot+1) = positionList(:,i_slot)+Doffset;
                i_slot = i_slot + 1;
            end                    
        else
            while positionList(1, i_slot) > edge_points_(1, i_edge_pos+1) && (i_slot <= params.time.nSlotsTotal-1)
                positionList(:,i_slot+1) = positionList(:,i_slot)+Doffset;
                i_slot = i_slot + 1;
            end
        end
        i_edge_pos = i_edge_pos + 1;
    end
    % filling of slots if num_slots > edge points
    if (i_edge_pos == size(edge_points_,2)) && (i_slot ~= params.time.nSlotsTotal-1)
        positionList(:,i_slot:end) = positionList(:,i_slot) .* ones(1, params.time.nSlotsTotal - i_slot+1);
        i_slot = params.time.nSlotsTotal;
    end

    positions = [positionList; 1.5*ones(1, params.time.nSlotsTotal)];
    predefinedUser.positions                    = [0; 0; 0]; % this position will be overwritten by the movement positions
    predefinedUser.nRX                          = 1;
    predefinedUser.indoorDecision               = parameters.indoorDecision.Geometry;
    predefinedUser.losDecision                  = parameters.losDecision.Geometry; % match random LOS decision with path loss model
    predefinedUser.userMovement.type            = parameters.setting.UserMovementType.Predefined;
    predefinedUser.userMovement.positionList    = positions;
    predefinedUser.channelModel                 = parameters.setting.ChannelModel.Quadriga;
    predefinedUser.technology                   = parameters.setting.NetworkElementTechnology.NRMN_5G;
    params.userParameters('predefinedMovementUser') = predefinedUser; % add user to parameter list

    %% Voigt %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % calculate macroscopic fading for each time slot when 'true'
    params.time.setSegmentLengthToOne       = true; % default: false
    
    % set this to get dependent shadowing maps (the same shadowing map in each chunk)
    params.shadowFading.randGeneratorState  = rng;  % default: empty

    % define the resolution of the large-scale fading maps (optional)
    params.largeScaleFadingMap      = parameters.largeScaleFadingMap.LargeScaleFadingMap;
    params.largeScaleFadingMap.roi  = params.regionOfInterest;
    params.largeScaleFadingMap.xRes = 50;   % x resolution in m
    params.largeScaleFadingMap.yRes = 50;   % y resolution in m
    params.largeScaleFadingMap.zRes = -1;   % z resolution in m (or "-1" -> only ground level)
    params.largeScaleFadingMap.on   = true; % default: 'false'

    
    % additional results
    params.save.lsfMap              = true;     % Calculate large-scale fading map ans save the result
    params.save.blockageMap         = true;     % Save wall blockage map
    params.save.isLosMap            = true;     % Save LOS/NLOS map
    params.save.isIndoorMap         = true;     % Save indoor/outdoor map
    params.save.pathLossMap         = true;     % Save path loss map
    params.save.wallLossMap         = true;     % Save wall Loss map
    params.save.shadowFadingMap     = true;     % Save shadow fading map
    params.save.antennaGainMap      = true;     % Save antenna gain map
    params.save.sirMap              = true;     % Save SIR map of each antenna
    params.save.combinedSirMap      = true;     % Save combined SIR map (highest SIR at each POI in the ROI)
    params.save.cellAssignmentMap   = true;     % Save optimal cell assignment map
    params.save.widebandSinrAllUsersdB  = true;
    params.save.receivePower        = true;
end

