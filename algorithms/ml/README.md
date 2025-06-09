## Common Machine Learning Algorithms

*   **Supervised Learning Algorithms:**
    *   **Linear Regression**: Used for regression tasks, predicting a continuous output based on a linear relationship between input features and the output.
    *   **Logistic Regression**: Used for binary classification tasks, predicting the probability of an instance belonging to a particular class. Despite its name, it's a classification algorithm.
    *   **Decision Trees**: Tree-like models that make decisions by recursively partitioning the data based on features. Can be used for both classification and regression.
    *   **Random Forests**: An ensemble learning method that constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
    *   **Support Vector Machines (SVMs)**: Powerful algorithms for classification and regression, finding an optimal hyperplane that best separates data points into different classes.
    *   **K-Nearest Neighbors (KNN)**: A non-parametric, instance-based learning algorithm used for both classification and regression. It classifies a data point based on the majority class among its 'k' nearest neighbors in the feature space.
    *   **Naive Bayes**: A family of probabilistic algorithms based on Bayes' theorem with the "naive" assumption of independence between features. Commonly used for text classification and spam detection.
    *   **Neural Networks**: Models inspired by the structure and function of the human brain, consisting of interconnected nodes (neurons) organized in layers. Capable of learning complex patterns and widely used for deep learning tasks.

*   **Unsupervised Learning Algorithms:**
    *   **K-Means Clustering**: An iterative algorithm that partitions 'n' observations into 'k' clusters, where each observation belongs to the cluster with the nearest mean (centroid).
    *   **Hierarchical Clustering**: Builds a hierarchy of clusters, either by starting with individual data points and merging them (agglomerative) or by starting with one large cluster and splitting it (divisive).
    *   **Principal Component Analysis (PCA)**: A popular technique for dimensionality reduction that transforms a dataset of possibly correlated variables into a set of linearly uncorrelated variables called principal components.
    *   **Independent Component Analysis (ICA)**: A computational method for separating a multivariate signal into additive subcomponents assuming the subcomponents are non-Gaussian signals and are statistically independent of each other.
    *   **Autoencoders**: Neural networks used for unsupervised learning of efficient data codings (representations). They learn to compress the input into a lower-dimensional code and then reconstruct the output from this code.
    *   **Gaussian Mixture Models (GMM)**: A probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. Used for clustering and density estimation.

*   **Reinforcement Learning Algorithms:**
    *   **Q-Learning**: A value-based reinforcement learning algorithm that learns an optimal policy by estimating the maximum expected future rewards for actions in given states.
    *   **SARSA (State-Action-Reward-State-Action)**: Similar to Q-learning, but it's an "on-policy" learning algorithm, meaning it learns the Q-value based on the current policy.
    *   **Deep Q Networks (DQN)**: Extends Q-learning by using deep neural networks to approximate the Q-value function, enabling it to handle complex environments with large state spaces.
    *   **Policy Gradients**: A family of algorithms that directly optimize the policy function (which maps states to actions) to maximize the expected reward.
    *   **Actor-Critic Methods**: Combine value-based (critic) and policy-based (actor) approaches. The actor learns the policy, while the critic estimates the value function, guiding the actor's learning.