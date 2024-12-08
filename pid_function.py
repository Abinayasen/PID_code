def pid_optimization_step_with_feedback(parameters, gradients, integral_error, prev_error, momentum_buffer, Kp, Ki, Kd, learning_rate):
    current_error = sum([(p.grad ** 2).sum().item() for p in parameters])  # Sum of squared gradients as the error

    # Compute PID terms
    proportional = current_error
    integral_error += current_error
    derivative = current_error - prev_error

    # Update learning rate using PID control
    if current_error > prev_error:
        learning_rate += (Kp * proportional + Ki * integral_error + Kd * derivative)
    else:
        learning_rate -= (Kp * proportional + Ki * integral_error + Kd * derivative)

    # Clamp the learning rate to stay within specified bounds
    learning_rate = max(0.0001, min(learning_rate, 1))

    # Update parameters with momentum
    for param, grad in zip(parameters, gradients):
        if param not in momentum_buffer:
            momentum_buffer[param] = torch.zeros_like(param)
        momentum_buffer[param] = momentum_buffer[param] * momentum + learning_rate * grad
        param.data -= momentum_buffer[param]

    return integral_error, current_error, learning_rate
