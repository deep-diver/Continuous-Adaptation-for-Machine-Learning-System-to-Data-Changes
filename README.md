# Continuous Adaptation for Machine Learning System to Data Changes

![](figures/overview.jpeg)

MLOps system evolves according to the changes of the world, and that is usually caused by [data/concept drift](https://en.wikipedia.org/wiki/Concept_drift). This project shows how to combine two separate pipelines, one for batch prediction and the other for training to adapt to data changes. 
- MLOps system also can be evolved when much better algorithm(i.e. SOTA model) comes out. In that case, the system should apply a better algorithm to understand the existing data better. We have demonstrated such workflow in [Model Training as a CI/CD System Part1: Reflect changes in codebase to MLOps pipeline](https://github.com/deep-diver/Model-Training-as-a-CI-CD-System) and [Model Training as a CI/CD System Part2: Trigger, schedule, and run MLOps pipelines](https://github.com/sayakpaul/CI-CD-for-Model-Training) projects.
