# Development notes

Keep the project straightforward. Prefer small functions with clear names over
clever abstractions. When changing model inputs, update the feature order in
both the training script and prediction endpoint, then add or adjust tests.

A few decisions worth understanding before presenting this project:

- The training script evaluates both models on the same held-out split.
- The API validates geographic inputs against the California bounds used in
  the interface.
- The saved prediction table is a simple history for the analytics screen.
- The range on the result screen is a display aid, not a confidence interval.
- The dataset describes census block groups and historical median values.
