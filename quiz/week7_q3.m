# Lecture 7 Quiz - Problem 3
#

w_xh = 0.5
w_hh = -1.0
w_hy = -0.7
h_bias = -1.0
y_bias = 0.0
x_inputs = [9, 4, -2]


function [logistic_unit_output] = logistic_activation(k)

  logistic_unit_output = 1 / ( 1 + exp( -1 * k ) );

endfunction


function [unit_logit] = get_logit(x_input, w_xh, h_input, w_hh, h_bias)

  unit_logit = x_input * w_xh + h_input * w_hh + h_bias;

endfunction


function [h_outputs, y_outputs] = get_rnn_outputs(x_inputs, w_xh, w_hh, w_hy, h_bias, y_bias)
  % Given a [1xN] column matrix of inputs, produce the
  % Recurrent Neural Network computation

  num_time_steps = size(x_inputs, 2);
  h_inputs = zeros(1, num_time_steps);
  h_logits = zeros(1, num_time_steps);
  h_outputs = zeros(1, num_time_steps);
  y_outputs = zeros(1, num_time_steps);

  % unlike the diagram, T here is 1:3, not 0:2
  for T = 1:num_time_steps

    printf('\nT %d:\n----------------\n', T);
    x_input = x_inputs(:, T)
    h_input = h_inputs(:, T)

    [unit_logit] = get_logit(x_input, w_xh, h_input, w_hh, h_bias)
    h_logits(1, T) = unit_logit;

    [logistic_unit_output] = logistic_activation(unit_logit)
    h_outputs(1, T) = logistic_unit_output;

    if T+1 <= num_time_steps
      % in all iterations except for the last,
      % make the hidden input at T+1 equal to the hidden output at T
      h_inputs(1, T+1) = h_outputs(1, T);
    endif

    y_output = logistic_unit_output * w_hy + y_bias
    y_outputs(1, T) = y_output;

  endfor

endfunction


[h_outputs, y_outputs] = get_rnn_outputs(x_inputs, w_xh, w_hh, w_hy, h_bias, y_bias);

printf("\nWhat is the value of the output y at T = 1? \n%f\n", y_outputs(1, 2))
printf("\nWhat is the value of the hidden state h at T = 2? \n%f\n", h_outputs(1,3))