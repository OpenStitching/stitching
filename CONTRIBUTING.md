# Contributing

All code contributions and bug reports are welcome!

Submit a new issue only if you are sure it is a missing feature or a bug.
For questions or if you are unsure [visit our discussions](https://github.com/OpenStitching/stitching/discussions).

Issues for newcomers are tagged with
['good first issue'](https://github.com/OpenStitching/stitching/labels/good%20first%20issue)
and documentation issues are tagged with
['documentation'](https://github.com/OpenStitching/stitching/labels/documentation).

We would be happy about co-maintainers, so do not hesitate contacting us
if you see yourself helping to continue this project!

## We love pull requests. Here's a quick guide

1. [Fork the repo](https://help.github.com/articles/fork-a-repo)
and create a branch for your new feature or bug fix.

2. Run the tests. We only take pull requests with passing tests: `python -m unittest`

3. Add at least one test for your change. Only refactoring and documentation changes
require no new tests.
Also make sure you submit a change specific to exactly one issue.
If you have ideas for multiple changes please create separate pull requests.

4. Make the test(s) pass.

5. Push to your fork and
[submit a pull request](https://help.github.com/articles/using-pull-requests).
A button should appear on your fork its github page afterwards.

## License Agreement

All contributions like pull requests, bug fixes, documentation changes and translations
fall under the Apache License and contributors agree to our
[contributor covenant code of conduct](https://github.com/OpenStitching/stitching/blob/main/CODE_OF_CONDUCT.md).

## Code formatting

We use the formatters and linters stated in
[.pre-commit-config.yaml](https://github.com/OpenStitching/stitching/blob/main/.pre-commit-config.yaml).
We use [pre-commit.ci](https://pre-commit.ci/)
to apply and enforce the formatting rules.
They are applied on your changes automatically once you open a Pull Request,
so you don't have to bother with formatting. Thanks pre-commit <3
