# Development


## Overview

Brieflow is meant to be very adaptable for new screening contexts.
As OPS develops, we expect users to make changes for their specific context that may eventually be included in the `main` branch of brieflow.

The general process of screen adaptation is as follows:

1) Make a new fork/branch for a new screening context.
Adapt Brieflow for this screening context.
2) If there is code that would be useful to the `main` branch, make a PR to contribute this code.
See the *Contributing* section below for more info.

[Matteo](https://github.com/mat10d) and [Roshan](https://github.com/roshankern) will be responsible for `main` brieflow development. 
These responsibilities include:

- Determining standards for OPS screening (where applicable)
- Updating documentation
- Helping users with issues
- Reviewing PRs to incorporate branches into the `main` branch


## Screen Adaptation

Users should do the following as they adapt Brieflow to their own screen:

1) Make a user-specific [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) of brieflow.
2) Make a screening context-specific [branch](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository) with the screening context code changes.
Make sure to self-review the PR to ensure the additional code is only what is useful for the main branch.

**Notes:**
- Follow the [Brieflow contribution notes](https://github.com/cheeseman-lab/brieflow/tree/updates-docs-2#contribution-notes) for code make new computational methodologies easier to integrate
- Follow the conventions below for making changes to Brieflow code for new computational functionality:
    - Module: Large changes to an existing module or extending Brieflow functionality
        - Changes to module env, rules, scripts, targets, functions
        - Ex: Adding a “Pipeline Evaluation” module to the end of Brieflow to compute metrics for an entire run
    - Process: Large change to an existing process script or adding a process script
        - Changes to process script, functions and possible changes to module env, rules, targets
        - Ex: Adding a script for a new type of illumination correction that can be used interchangeably with old script
    - Function: Small changes to existing function
        - Changes to function code in Brieflow library
        - Ex: adding an option to `extract_nd2_metadata` to extract an additional piece of metadata


## Contributing

We aim to strike a balance between simplicity and funcitonaliy in the `main` branch.
For this reason, we will only incorporate code into main that is helpful for a general/standardized OPS screening context.
If you have code that would help other brieflow users, please make a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) to `main`.
Matteo and Roshan will review this pull request and guide it to be merged!
