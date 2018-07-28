function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.

    % 1. The variation that we're using here is the one where every time after calculating a conditional probability for a unit, 
    % we sample a state for the unit from that conditional probability (using the functionsample_bernoulli)
    visible_state = sample_bernoulli(visible_data);
    % size <number of visible units> by <number of data cases>

    % 2. we'll sample a binary state for the hidden units conditional on the data;
    % We use it once on the given data and the hidden state that it gives rise to. 
    % That gives us the direction of changing the weights that will make the data have greater goodness, 
    % which is what we want to achieve.
    hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state);
    hidden_state = sample_bernoulli(hidden_probability);
    % size <number of hidden units> by <number of data cases>

    d_G_1 = configuration_goodness_gradient(visible_state, hidden_state);


    % 3. We also use it on the "reconstruction" visible state and the hidden state that it gives rise to. 
    % That gives us the direction of changing the weights that will make the reconstruction have greater goodness, 
    % so we want to go in the opposite direction, because we want to make the reconstruction have less goodness.
    visible_probability = hidden_state_to_visible_probabilities(rbm_w, hidden_state);
    visible_state_2 = sample_bernoulli(visible_probability);

    hidden_probability_2 = visible_state_to_hidden_probabilities(rbm_w, visible_state_2);
    hidden_state_2 = sample_bernoulli(hidden_probability_2);

    d_G_2 = configuration_goodness_gradient(visible_state_2, hidden_state_2);

    ret = d_G_1 - d_G_2;
end
