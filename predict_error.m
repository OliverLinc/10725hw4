function [error] = predict_error(y, y_predict)
%UNTITLED14 Summary of this function goes here
%   Detailed explanation goes here
error = sum(abs(y-y_predict))/length(y);
end

