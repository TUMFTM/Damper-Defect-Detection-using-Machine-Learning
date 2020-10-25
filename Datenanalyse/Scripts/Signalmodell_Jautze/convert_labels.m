function [is_defect] = convert_labels(data_for_analysis)
%CONVERT_LABELS Summary of this function goes here
%   Detailed explanation goes here
    is_defect = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);

    is_defect.SPEED_FL = zeros(size(data_for_analysis.Prop.labelIsolation,1),1);
    is_defect.SPEED_FL(logical(data_for_analysis.Prop.labelIsolation == "passiveIntact")) = 0;
    is_defect.SPEED_FL(logical(data_for_analysis.Prop.labelIsolation == "allDampersDefect")) = 1;
    is_defect.SPEED_FL(logical(data_for_analysis.Prop.labelIsolation == "FLDamperDefect")) = 1;
    is_defect.SPEED_FL(logical(data_for_analysis.Prop.labelIsolation == "FR")) = 0;
    is_defect.SPEED_FL(logical(data_for_analysis.Prop.labelIsolation == "RL")) = 0;
    is_defect.SPEED_FL(logical(data_for_analysis.Prop.labelIsolation == "RRDamperDefect")) = 0;

    is_defect.SPEED_FR = zeros(size(data_for_analysis.Prop.labelIsolation,1),1);
    is_defect.SPEED_FR(logical(data_for_analysis.Prop.labelIsolation == "passiveIntact")) = 0;
    is_defect.SPEED_FR(logical(data_for_analysis.Prop.labelIsolation == "allDampersDefect")) = 1;
    is_defect.SPEED_FR(logical(data_for_analysis.Prop.labelIsolation == "FLDamperDefect")) = 0;
    is_defect.SPEED_FR(logical(data_for_analysis.Prop.labelIsolation == "FR")) = 1;
    is_defect.SPEED_FR(logical(data_for_analysis.Prop.labelIsolation == "RL")) = 0;
    is_defect.SPEED_FR(logical(data_for_analysis.Prop.labelIsolation == "RRDamperDefect")) = 0;

    is_defect.SPEED_RL = zeros(size(data_for_analysis.Prop.labelIsolation,1),1);
    is_defect.SPEED_RL(logical(data_for_analysis.Prop.labelIsolation == "passiveIntact")) = 0;
    is_defect.SPEED_RL(logical(data_for_analysis.Prop.labelIsolation == "allDampersDefect")) = 1;
    is_defect.SPEED_RL(logical(data_for_analysis.Prop.labelIsolation == "FLDamperDefect")) = 0;
    is_defect.SPEED_RL(logical(data_for_analysis.Prop.labelIsolation == "FR")) = 0;
    is_defect.SPEED_RL(logical(data_for_analysis.Prop.labelIsolation == "RL")) = 1;
    is_defect.SPEED_RL(logical(data_for_analysis.Prop.labelIsolation == "RRDamperDefect")) = 0;

    is_defect.SPEED_RR = zeros(size(data_for_analysis.Prop.labelIsolation,1),1);
    is_defect.SPEED_RR(logical(data_for_analysis.Prop.labelIsolation == "passiveIntact")) = 0;
    is_defect.SPEED_RR(logical(data_for_analysis.Prop.labelIsolation == "allDampersDefect")) = 1;
    is_defect.SPEED_RR(logical(data_for_analysis.Prop.labelIsolation == "FLDamperDefect")) = 0;
    is_defect.SPEED_RR(logical(data_for_analysis.Prop.labelIsolation == "FR")) = 0;
    is_defect.SPEED_RR(logical(data_for_analysis.Prop.labelIsolation == "RL")) = 0;
    is_defect.SPEED_RR(logical(data_for_analysis.Prop.labelIsolation == "RRDamperDefect")) = 1;
end

