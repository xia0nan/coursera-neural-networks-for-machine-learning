# Lecture 7 Quiz - Problem 4
#

w_xh = -0.1
w_hh = 0.5
w_hy = 0.25
h_bias = 0.4
y_bias = 0.0
x_inputs = [18, 9, -8];
t_outputs = [0.1, -0.1, -0.2]


function [logistic_unit_output] = logistic_activation(k)

  logistic_unit_output = 1 / ( 1 + exp( -1 * k ) );

endfunction


function [unit_logit] = get_logit(x_input, w_xh, h_input, w_hh, h_bias)

  unit_logit = x_input * w_xh + h_input * w_hh + h_bias;

endfunction


function [h_logits, h_outputs, y_outputs] = get_rnn_outputs(x_inputs, w_xh, w_hh, w_hy, h_bias, y_bias)
  % Given a [1xN] column matrix of inputs, produce the
  % Recurrent Neural Network computation

  num_time_steps = size(x_inputs, 2);
  h_inputs = zeros(1, num_time_steps);
  h_logits = zeros(1, num_time_steps);
  h_outputs = zeros(1, num_time_steps);
  y_outputs = zeros(1, num_time_steps);

  % unlike the diagram, T here is 1:3, not 0:2
  for T = 1:num_time_steps

    % printf('\nT %d:\n----------------\n', T);
    x_input = x_inputs(:, T);
    h_input = h_inputs(:, T);

    [unit_logit] = get_logit(x_input, w_xh, h_input, w_hh, h_bias);
    h_logits(1, T) = unit_logit;

    [logistic_unit_output] = logistic_activation(unit_logit);
    h_outputs(1, T) = logistic_unit_output;

    if T+1 <= num_time_steps
      % in all iterations except for the last,
      % make the hidden input at T+1 equal to the hidden output at T
      h_inputs(1, T+1) = h_outputs(1, T);
    endif

    y_output = logistic_unit_output * w_hy + y_bias;
    y_outputs(1, T) = y_output;

  endfor

endfunction

function [error_val] = get_squared_error_loss(y, t)
  error_val = sum(0.5*(t-y).^2);
endfunction


[h_logits, h_outputs, y_outputs] = get_rnn_outputs(x_inputs, w_xh, w_hh, w_hy, h_bias, y_bias);
[error_val] = get_squared_error_loss(y_outputs, t_outputs)

% disp("hidden inputs:"), disp(x_inputs)
% disp("hidden logits:"), disp(h_logits)
% disp("hidden outputs:"), disp(h_outputs)
% disp("y outputs:"), disp(y_outputs)
% disp("error amount"), disp(error_val)

function [dE1dz1] = get_dE1dz1(y_1, t_1, w_hy, h_1)
  dE1dy1 = y_1 - t_1;
  dy1dh1 = w_hy;
  dh1dz1 = h_1 * (1 - h_1);
  dE1dz1 = dE1dy1 * dy1dh1 * dh1dz1;
endfunction

function [dE2dz1] = get_dE2dz1(y_2, t_2, w_hy, h_2, w_hh, h_1)
  dE2dy2 = y_2 - t_2;
  dy2dh2 = w_hy;
  dh2dz2 = h_2 * ( 1 - h_2 );
  dz2dh1 = w_hh;
  dh1dz1 = h_1 * ( 1 - h_1 );
  dE2dz1 = dE2dy2 * dy2dh2 * dh2dz2 * dz2dh1 * dh1dz1;
endfunction


y_1 = y_outputs(:, 2)
y_2 = y_outputs(:, 3)

t_1 = t_outputs(:, 2)
t_2 = t_outputs(:, 3)

h_1 = h_outputs(:, 2)
h_2 = h_outputs(:, 3)

dE1dz1 = get_dE1dz1(y_1, t_1, w_hy, h_1)
dE2dz1 = get_dE2dz1(y_2, t_2, w_hy, h_2, w_hh, h_1)

dEdz1 = dE1dz1 + dE2dz1