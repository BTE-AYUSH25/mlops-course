

## OOPs Fundamentals for Data Scientists

### Why OOPs for ML Engineers?

| Traditional Approach | OOPs Approach | Benefit |
|---------------------|---------------|---------|
| Procedural code in Jupyter | Modular, reusable classes | Production-ready code |
| Script-based ML | Class-based ML pipelines | Better organization |
| Global variables | Encapsulated data | Data security |
| Function spaghetti | Method organization | Easy debugging |

## Core OOPs Concepts

### Class vs Object

| Concept | Definition | Real-world Example | Python Example |
|---------|------------|-------------------|----------------|
| **Class** | Blueprint/template | Car design blueprint | `class Car:` |
| **Object** | Instance of class | Actual car from blueprint | `my_car = Car()` |
| **Attributes** | Characteristics/properties | Color, engine, model | `self.color = "red"` |
| **Methods** | Actions/behaviors | Drive, brake, honk | `def drive(self):` |

### Basic Class Implementation

```python
class Employee:
    # Class attribute (shared by all instances)
    company = "Tech Corp"
    
    def __init__(self, name, salary, designation):
        # Instance attributes (unique to each object)
        self.name = name
        self.salary = salary
        self.designation = designation
    
    # Method
    def report(self):
        return f"{self.name} is working as {self.designation}"

# Creating objects
emp1 = Employee("Sam Altman", 100000, "CEO")
emp2 = Employee("John Doe", 75000, "Data Scientist")

print(emp1.report())  # Output: Sam Altman is working as CEO
```

## The Four Pillars of OOPs

### 1. Encapsulation

| Concept | Purpose | Implementation | Example |
|---------|---------|----------------|---------|
| **Data Hiding** | Protect sensitive data | `__` (double underscore) | `__salary` |
| **Getter** | Access private data | `get_salary()` method | `def get_salary(self):` |
| **Setter** | Modify private data | `set_salary()` method | `def set_salary(self, amount):` |

```python
class BankAccount:
    def __init__(self, account_holder, balance):
        self.account_holder = account_holder
        self.__balance = balance  # Private attribute
    
    # Getter method
    def get_balance(self):
        return self.__balance
    
    # Setter method with validation
    def set_balance(self, amount):
        if amount >= 0:
            self.__balance = amount
        else:
            print("Balance cannot be negative")
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return f"Deposited ${amount}"
        return "Invalid deposit amount"

# Usage
account = BankAccount("Alice", 1000)
print(account.get_balance())  # 1000
account.set_balance(1500)
print(account.get_balance())  # 1500
```

### 2. Inheritance

#### Inheritance Types

| Type | Description | Syntax | Use Case |
|------|-------------|--------|----------|
| **Single** | One parent, one child | `class Child(Parent):` | Basic inheritance |
| **Multi-level** | Chain of inheritance | `class GrandChild(Child):` | Hierarchical systems |
| **Hierarchical** | Multiple children, one parent | `class Child1(Parent): class Child2(Parent):` | Different categories |
| **Multiple** | One child, multiple parents | `class Child(Parent1, Parent2):` | Feature combination |

```python
# Parent Class
class MLModel:
    def __init__(self, model_name, accuracy):
        self.model_name = model_name
        self.accuracy = accuracy
        self.__trained = False  # Private attribute
    
    def train(self):
        self.__trained = True
        return f"{self.model_name} is training..."
    
    def predict(self, data):
        if self.__trained:
            return f"Predicting with {self.model_name}"
        return "Model needs training first"

# Child Class - Single Inheritance
class LinearModel(MLModel):
    def __init__(self, model_name, accuracy, coefficients):
        super().__init__(model_name, accuracy)  # Call parent constructor
        self.coefficients = coefficients
    
    # Method overriding
    def train(self):
        result = super().train()  # Call parent method
        return f"Linear Model: {result}"
    
    # Child-specific method
    def get_coefficients(self):
        return self.coefficients

# Child Class - Multiple Inheritance
class Preprocessor:
    def __init__(self, technique):
        self.technique = technique
    
    def preprocess(self, data):
        return f"Preprocessing with {self.technique}"

class AdvancedModel(MLModel, Preprocessor):
    def __init__(self, model_name, accuracy, technique):
        MLModel.__init__(self, model_name, accuracy)
        Preprocessor.__init__(self, technique)
    
    def full_pipeline(self, data):
        processed = self.preprocess(data)
        prediction = self.predict(processed)
        return f"{processed} -> {prediction}"

# Usage
linear_model = LinearModel("Linear Regression", 0.85, [1.2, -0.5])
print(linear_model.train())  # Linear Model: Linear Regression is training...

advanced_model = AdvancedModel("Random Forest", 0.92, "Standard Scaling")
print(advanced_model.full_pipeline("raw_data"))
```

### 3. Polymorphism

| Type | Description | Example |
|------|-------------|---------|
| **Method Overriding** | Same method name, different implementation in child class | `train()` method in parent vs child |
| **Method Overloading** | Same method name, different parameters | Python doesn't support directly |
| **Duck Typing** | Object's type determined by methods it has | "If it quacks like a duck..." |

```python
class Classifier:
    def classify(self):
        return "Generic classification"

class RandomForest(Classifier):
    def classify(self):
        return "Random Forest classification"

class SVM(Classifier):
    def classify(self):
        return "SVM classification"

# Polymorphism in action
def perform_classification(classifier):
    return classifier.classify()

# Same method, different behaviors
models = [Classifier(), RandomForest(), SVM()]
for model in models:
    print(perform_classification(model))
```

### 4. Abstraction

```python
from abc import ABC, abstractmethod

class MLAlgorithm(ABC):
    @abstractmethod
    def train(self, data):
        pass
    
    @abstractmethod
    def predict(self, input_data):
        pass
    
    # Concrete method
    def evaluate(self, test_data):
        predictions = self.predict(test_data)
        return f"Evaluating with {predictions}"

class NeuralNetwork(MLAlgorithm):
    def train(self, data):
        return "Training neural network with backpropagation"
    
    def predict(self, input_data):
        return "Neural network prediction"

# This will raise error without implementing abstract methods
# algorithm = MLAlgorithm()  # Error!

nn = NeuralNetwork()
print(nn.train("data"))  # Training neural network with backpropagation
```

## Advanced OOPs Concepts

### Constructors and Magic Methods

| Method | Purpose | Called When | Example |
|--------|---------|-------------|---------|
| `__init__` | Initialize object | Object creation | `obj = Class()` |
| `__str__` | String representation | `print(obj)` | `def __str__(self):` |
| `__len__` | Length of object | `len(obj)` | `def __len__(self):` |
| `__call__` | Make object callable | `obj()` | `def __call__(self):` |

```python
class Dataset:
    def __init__(self, data, name):
        self.data = data
        self.name = name
        print(f"Dataset '{name}' initialized automatically!")
    
    def __str__(self):
        return f"Dataset: {self.name} with {len(self.data)} samples"
    
    def __len__(self):
        return len(self.data)
    
    def __call__(self):
        return f"Dataset {self.name} is being used"

# Magic methods in action
dataset = Dataset([1, 2, 3, 4, 5], "Training Data")  # __init__ called automatically
print(dataset)    # __str__ called
print(len(dataset))  # __len__ called
print(dataset())  # __call__ called
```

### Static Methods and Variables

| Type | Belongs To | Access Via | Has `self`? |
|------|------------|------------|-------------|
| **Instance Method** | Object | `object.method()` | Yes |
| **Class Method** | Class | `Class.method()` or `object.method()` | `cls` instead of `self` |
| **Static Method** | Class | `Class.method()` or `object.method()` | No |

```python
class MLExperiment:
    # Class variable (shared across all instances)
    experiment_count = 0
    
    def __init__(self, name):
        self.name = name
        MLExperiment.experiment_count += 1
        self.experiment_id = MLExperiment.experiment_count
    
    # Instance method
    def get_details(self):
        return f"Experiment {self.experiment_id}: {self.name}"
    
    # Class method
    @classmethod
    def get_total_experiments(cls):
        return f"Total experiments: {cls.experiment_count}"
    
    # Static method
    @staticmethod
    def validate_name(name):
        if len(name) < 3:
            return "Name too short"
        return "Valid name"

# Usage
exp1 = MLExperiment("Random Forest Tuning")
exp2 = MLExperiment("Neural Network Architecture")

print(exp1.get_details())  # Instance method
print(MLExperiment.get_total_experiments())  # Class method
print(MLExperiment.validate_name("AB"))  # Static method
```

## OOPs for ML Projects

### ML Pipeline with OOPs

```python
class DataPreprocessor:
    def __init__(self, scaling_method="standard"):
        self.scaling_method = scaling_method
        self.__is_fitted = False
    
    def fit(self, data):
        self.__is_fitted = True
        return f"Fitted {self.scaling_method} scaler"
    
    def transform(self, data):
        if not self.__is_fitted:
            return "Preprocessor needs fitting first"
        return f"Transforming data with {self.scaling_method}"

class Model(MLModel):  # Inheriting from earlier class
    def __init__(self, model_name, preprocessor):
        super().__init__(model_name, accuracy=0.0)
        self.preprocessor = preprocessor
    
    def full_training(self, raw_data):
        # Use composition
        self.preprocessor.fit(raw_data)
        processed_data = self.preprocessor.transform(raw_data)
        training_result = self.train()
        return f"{processed_data} -> {training_result}"

# Complete ML Pipeline
preprocessor = DataPreprocessor("min-max")
model = Model("Gradient Boosting", preprocessor)
print(model.full_training("raw_dataset"))
```

### Model Versioning with OOPs

```python
class ModelVersion:
    version_counter = 0
    
    def __init__(self, model_type, parameters):
        ModelVersion.version_counter += 1
        self.version = f"v{ModelVersion.version_counter}.0"
        self.model_type = model_type
        self.parameters = parameters
        self.__training_data = None
    
    def set_training_data(self, data):
        self.__training_data = data
    
    def get_model_info(self):
        return {
            "version": self.version,
            "type": self.model_type,
            "parameters": self.parameters
        }
    
    def __str__(self):
        return f"Model {self.version} ({self.model_type})"

# Version management
models = [
    ModelVersion("RandomForest", {"n_estimators": 100}),
    ModelVersion("XGBoost", {"learning_rate": 0.1}),
    ModelVersion("NeuralNetwork", {"layers": [100, 50, 10]})
]

for model in models:
    print(model)
```

## Memory Aids & Quick Reference

### 🎯 OOPs Memory Aids (Hindi & English)

| Concept | Memory Aid (English) | याद रखने का तरीका (Hindi) |
|---------|---------------------|--------------------------|
| **Class** | "Blueprint for a house" | "मकान का नक्शा" |
| **Object** | "Actual house from blueprint" | "नक्शे से बना असली मकान" |
| **Attributes** | "House properties (color, rooms)" | "मकान के गुण (रंग, कमरे)" |
| **Methods** | "House actions (open door, turn lights)" | "मकान के काम (दरवाज़ा खोलना, लाइट चालू करना)" |
| **Constructor** | "Automatic setup when house built" | "मकान बनते ही अपने आप होने वाला काम" |
| **Self** | "This specific house's identity" | "इस specific मकान की पहचान" |
| **Encapsulation** | "Locked safe inside house" | "मकान के अंदर तिजोरी जिसमें महत्वपूर्ण चीज़ें" |
| **Inheritance** | "Child gets parent's features" | "बच्चे को माता-पिता के गुण मिलना" |
| **Polymorphism** | "Same button, different actions" | "एक बटन, अलग-अलग काम" |
| **Static** | "Shared family property" | "परिवार की साझा संपत्ति" |

### 🔑 Key Syntax Quick Reference

```python
# Class Definition
class ClassName:
    class_var = "value"  # Class variable
    
    def __init__(self, param):  # Constructor
        self.instance_var = param  # Instance variable
        self.__private_var = "hidden"  # Private variable
    
    def method(self):  # Instance method
        return self.instance_var
    
    @classmethod
    def class_method(cls):  # Class method
        return cls.class_var
    
    @staticmethod
    def static_method():  # Static method
        return "No self/cls"

# Inheritance
class ChildClass(ParentClass):
    def __init__(self, param):
        super().__init__(param)  # Call parent constructor

# Encapsulation
obj._protected_var    # Protected (convention)
obj.__private_var     # Private (name mangling)

# Magic Methods
def __str__(self): return "string representation"
def __len__(self): return length
def __call__(self): return "callable object"
```

### 🚀 Daily OOPs Practice Patterns

```python
# Pattern 1: Basic Class
class DataScientist:
    def __init__(self, name, skills):
        self.name = name
        self.skills = skills
    
    def display_skills(self):
        return f"{self.name} knows: {', '.join(self.skills)}"

# Pattern 2: Inheritance
class MLEngineer(DataScientist):
    def __init__(self, name, skills, frameworks):
        super().__init__(name, skills)
        self.frameworks = frameworks

# Pattern 3: Encapsulation
class SensitiveData:
    def __init__(self, data):
        self.__encrypted_data = self.__encrypt(data)
    
    def __encrypt(self, data):
        return f"encrypted_{data}"

# Pattern 4: Composition
class MLPipeline:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
```

## Common OOPs Mistakes & Solutions

| Mistake | Problem | Solution |
|---------|---------|----------|
| Forgetting `self` | Methods can't access instance data | Always include `self` as first parameter |
| Not calling `super()` | Parent constructor not initialized | Use `super().__init__()` in child classes |
| Direct access to private vars | Breaks encapsulation | Use getter/setter methods |
| Overusing inheritance | Complex hierarchy | Prefer composition when possible |
| Ignoring magic methods | Poor object representation | Implement `__str__`, `__repr__` |

---

## 🎯 Next Steps for MLOps with OOPs

1. **Convert your ML scripts** to class-based structure
2. **Create ML pipeline classes** for data loading, preprocessing, training
3. **Implement model versioning** using OOPs principles
4. **Practice encapsulation** for sensitive model parameters
5. **Use inheritance** for different model types

*"OOPs turns ML scripts into production-ready systems!"* 🚀
```

**Also create: `day-3-oops/oops-quick-reference.md`**

```markdown
# OOPs Quick Reference Card

## Basic Syntax
```python
class ClassName:
    class_var = "shared"          # Class variable
    
    def __init__(self, param):    # Constructor
        self.inst_var = param     # Instance variable
        self.__priv_var = "hid"   # Private variable
    
    def method(self):             # Instance method
        return self.inst_var
    
    @classmethod
    def cls_method(cls):          # Class method
        return cls.class_var
    
    @staticmethod
    def static_method():          # Static method
        return "No self"
```

## Inheritance
```python
class Parent:
    def __init__(self, name):
        self.name = name

class Child(Parent):
    def __init__(self, name, skill):
        super().__init__(name)    # Call parent constructor
        self.skill = skill
```

## Encapsulation
```python
class SecureClass:
    def __init__(self, data):
        self.__private_data = data
    
    def get_data(self):           # Getter
        return self.__private_data
    
    def set_data(self, data):     # Setter
        if data: self.__private_data = data
```

## Magic Methods
```python
def __str__(self):    # print(obj)
    return "Readable string"

def __len__(self):    # len(obj)
    return length

def __call__(self):   # obj()
    return "Called!"
```

## Pillars Summary
- **Encapsulation**: Hide data with `__`
- **Inheritance**: `class Child(Parent)`
- **Polymorphism**: Same method, different behavior
- **Abstraction**: `ABC` and `@abstractmethod`
```

## Next Steps:

1. **Save these files** in your `day-3-oops/` directory
2. **Add and commit** them to your repository:
```bash
git add day-3-oops/
git commit -m "docs: add comprehensive OOPs guide with memory aids and examples"
git push origin main
```

3. **Practice** the concepts with your ML code
4. **Convert** existing scripts to OOPs style

You're absolutely right! Let me add a comprehensive section on all inheritance types with detailed examples:

**Update the Inheritance section in `day-3-oops/oops-complete-guide.md`:**

```markdown
## Complete Inheritance Types with Examples

### 1. Single Inheritance
One child class inherits from one parent class.

```python
class MLModel:
    """Parent Class - Base ML Model"""
    def __init__(self, name):
        self.name = name
        self.is_trained = False
    
    def train(self):
        self.is_trained = True
        return f"{self.name} is training"
    
    def predict(self):
        if self.is_trained:
            return f"{self.name} making prediction"
        return "Train the model first"

class LinearRegression(MLModel):
    """Child Class - Single Inheritance"""
    def __init__(self, name, learning_rate):
        super().__init__(name)
        self.learning_rate = learning_rate
    
    # Method overriding
    def train(self):
        result = super().train()
        return f"Linear Regression: {result} with LR {self.learning_rate}"
    
    # Child-specific method
    def calculate_coefficients(self):
        return f"Calculating coefficients for {self.name}"

# Usage
lr_model = LinearRegression("Linear Model", 0.01)
print(lr_model.train())      # Linear Regression: Linear Model is training with LR 0.01
print(lr_model.predict())    # Linear Model making prediction
```

### 2. Multiple Inheritance
One child class inherits from multiple parent classes.

```python
class DataLoader:
    """First Parent Class"""
    def __init__(self, data_source):
        self.data_source = data_source
    
    def load_data(self):
        return f"Loading data from {self.data_source}"
    
    def validate_data(self):
        return "Data validation completed"

class Preprocessor:
    """Second Parent Class"""
    def __init__(self, method):
        self.method = method
    
    def preprocess(self):
        return f"Preprocessing with {self.method}"
    
    def clean_data(self):
        return "Data cleaning completed"

class CompletePipeline(DataLoader, Preprocessor):
    """Child Class - Multiple Inheritance"""
    def __init__(self, data_source, method, model_name):
        # Initialize both parent classes
        DataLoader.__init__(self, data_source)
        Preprocessor.__init__(self, method)
        self.model_name = model_name
    
    def run_pipeline(self):
        load_result = self.load_data()
        preprocess_result = self.preprocess()
        return f"{load_result} -> {preprocess_result} -> Training {self.model_name}"

# Usage
pipeline = CompletePipeline("database", "normalization", "Random Forest")
print(pipeline.run_pipeline())
print(pipeline.validate_data())  # From DataLoader
print(pipeline.clean_data())     # From Preprocessor
```

### 3. Multi-level Inheritance
Chain of inheritance - Grandparent → Parent → Child

```python
class BaseModel:
    """Grandparent Class"""
    def __init__(self):
        self.created_at = "2024-01-01"
        self.version = "1.0"
    
    def info(self):
        return f"Base Model v{self.version} created at {self.created_at}"
    
    def save_model(self):
        return "Model saved to disk"

class NeuralNetwork(BaseModel):
    """Parent Class"""
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.model_type = "Neural Network"
    
    def forward_pass(self, input_data):
        return f"Forward pass through {len(self.layers)} layers"
    
    def backward_pass(self):
        return "Backward pass for training"

class CNN(NeuralNetwork):
    """Child Class - Multi-level Inheritance"""
    def __init__(self, layers, kernel_size):
        super().__init__(layers)
        self.kernel_size = kernel_size
        self.cnn_specific = "Convolutional Operations"
    
    def apply_convolution(self):
        return f"Applying convolution with kernel {self.kernel_size}"
    
    # Override parent method
    def forward_pass(self, input_data):
        base_result = super().forward_pass(input_data)
        return f"CNN: {base_result} with convolution"

# Usage
cnn = CNN([32, 64, 128], 3)
print(cnn.info())              # From BaseModel (Grandparent)
print(cnn.forward_pass("img")) # From NeuralNetwork (Parent) - overridden
print(cnn.apply_convolution()) # From CNN (Child) - specific method
print(cnn.save_model())        # From BaseModel (Grandparent)
```

### 4. Hierarchical Inheritance
Multiple child classes inherit from one parent class.

```python
class FeatureEngineer:
    """Parent Class"""
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.engineered = False
    
    def engineer_features(self, data):
        self.engineered = True
        return f"Engineering {self.feature_name} features"
    
    def get_feature_importance(self):
        return f"Feature importance for {self.feature_name}"

class NumericalFeatureEngineer(FeatureEngineer):
    """First Child Class"""
    def __init__(self, feature_name, scaling_method):
        super().__init__(feature_name)
        self.scaling_method = scaling_method
    
    def scale_features(self):
        return f"Scaling numerical features using {self.scaling_method}"
    
    # Override parent method
    def engineer_features(self, data):
        base_result = super().engineer_features(data)
        return f"Numerical: {base_result} with scaling"

class CategoricalFeatureEngineer(FeatureEngineer):
    """Second Child Class"""
    def __init__(self, feature_name, encoding_method):
        super().__init__(feature_name)
        self.encoding_method = encoding_method
    
    def encode_features(self):
        return f"Encoding categorical features using {self.encoding_method}"
    
    # Override parent method
    def engineer_features(self, data):
        base_result = super().engineer_features(data)
        return f"Categorical: {base_result} with encoding"

# Usage
num_engineer = NumericalFeatureEngineer("age", "standard")
cat_engineer = CategoricalFeatureEngineer("city", "one-hot")

print(num_engineer.engineer_features("data"))  # Numerical: Engineering age features with scaling
print(num_engineer.scale_features())           # Scaling numerical features using standard

print(cat_engineer.engineer_features("data"))  # Categorical: Engineering city features with encoding
print(cat_engineer.encode_features())          # Encoding categorical features using one-hot

# Both can access parent method
print(num_engineer.get_feature_importance())   # From parent
print(cat_engineer.get_feature_importance())   # From parent
```

### 5. Hybrid Inheritance
Combination of multiple inheritance types.

```python
class Logger:
    """Mixin Class for logging"""
    def log(self, message):
        return f"LOG: {message}"

class BaseML:
    """Base ML Class"""
    def __init__(self, name):
        self.name = name
    
    def train(self):
        return f"Training {self.name}"

class SupervisedModel(BaseML, Logger):
    """Multiple Inheritance + Hierarchical"""
    def __init__(self, name, target_variable):
        BaseML.__init__(self, name)
        self.target_variable = target_variable
    
    def supervised_train(self):
        train_result = self.train()
        log_result = self.log(f"Training supervised model {self.name}")
        return f"{train_result} | {log_result}"

class ClassificationModel(SupervisedModel):
    """Multi-level + Multiple Inheritance"""
    def __init__(self, name, target_variable, classes):
        super().__init__(name, target_variable)
        self.classes = classes
    
    def classify(self):
        train_result = self.supervised_train()
        return f"{train_result} -> Classifying into {len(self.classes)} classes"

# Usage
classifier = ClassificationModel("SVM", "species", ["setosa", "versicolor", "virginica"])
print(classifier.classify())
# Training SVM -> Classifying into 3 classes
```

### 6. Diamond Problem in Multiple Inheritance

```python
class BaseClass:
    def __init__(self):
        print("BaseClass constructor")
    
    def method(self):
        return "BaseClass method"

class ClassA(BaseClass):
    def __init__(self):
        super().__init__()
        print("ClassA constructor")
    
    def method(self):
        return "ClassA method"

class ClassB(BaseClass):
    def __init__(self):
        super().__init__()
        print("ClassB constructor")
    
    def method(self):
        return "ClassB method"

class DiamondClass(ClassA, ClassB):
    def __init__(self):
        super().__init__()  # Python handles diamond problem with MRO
        print("DiamondClass constructor")
    
    def test_methods(self):
        return f"Method resolution: {self.method()}"

# Method Resolution Order (MRO)
print("MRO:", DiamondClass.__mro__)
# Output: MRO: (<class '__main__.DiamondClass'>, <class '__main__.ClassA'>, 
#                <class '__main__.ClassB'>, <class '__main__.BaseClass'>, <class 'object'>)

diamond = DiamondClass()
print(diamond.test_methods())  # Uses ClassA method due to MRO
```

### Method Resolution Order (MRO) Examples

```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

class E(C, B):
    pass

# Check MRO
print("D MRO:", [cls.__name__ for cls in D.__mro__])
# Output: D MRO: ['D', 'B', 'C', 'A', 'object']

print("E MRO:", [cls.__name__ for cls in E.__mro__])
# Output: E MRO: ['E', 'C', 'B', 'A', 'object']

# Usage
d_obj = D()
e_obj = E()

print("D method:", d_obj.method())  # B (because B comes before C in MRO)
print("E method:", e_obj.method())  # C (because C comes before B in MRO)
```

### Real-world ML Inheritance Example

```python
class DataValidator:
    def validate(self, data):
        return "Basic data validation"

class Model:
    def __init__(self, name):
        self.name = name
    
    def train(self, data):
        return f"Training {self.name}"

class Preprocessor:
    def preprocess(self, data):
        return "Data preprocessing"

class MLPipeline(DataValidator, Model, Preprocessor):
    """Combining multiple functionalities through inheritance"""
    def __init__(self, name):
        Model.__init__(self, name)
    
    def run_complete_pipeline(self, raw_data):
        # Use all inherited methods
        validation = self.validate(raw_data)
        preprocessing = self.preprocess(raw_data)
        training = self.train(preprocessing)
        
        return f"""
        ML Pipeline Results:
        - Validation: {validation}
        - Preprocessing: {preprocessing}
        - Training: {training}
        """

# Usage
pipeline = MLPipeline("Random Forest")
result = pipeline.run_complete_pipeline("raw_dataset")
print(result)
```

## Inheritance Summary Table

| Type | Structure | Example | Use Case |
|------|-----------|---------|----------|
| **Single** | Parent → Child | `LinearRegression(MLModel)` | Basic specialization |
| **Multiple** | Child(Parent1, Parent2) | `Pipeline(Loader, Processor)` | Combine functionalities |
| **Multi-level** | Grandparent → Parent → Child | `CNN(NeuralNetwork(BaseModel))` | Progressive specialization |
| **Hierarchical** | Parent → Child1, Child2 | `NumEngineer(Engineer), CatEngineer(Engineer)` | Different specializations from same base |
| **Hybrid** | Combination of above | Complex ML pipelines | Real-world complex systems |

## Inheritance Best Practices

1. **Use inheritance for "is-a" relationships** (CNN is a NeuralNetwork)
2. **Prefer composition over deep inheritance** for "has-a" relationships
3. **Keep inheritance hierarchy shallow** (2-3 levels max)
4. **Use mixins** for cross-cutting concerns (like Logging)
5. **Understand MRO** for multiple inheritance scenarios
6. **Document inheritance relationships** clearly

This complete inheritance section covers all types with practical ML examples and addresses the diamond problem with MRO explanations!
```

**Also update the quick reference card:**

```markdown
# Inheritance Types Quick Reference

## Single Inheritance
```python
class Child(Parent):
    def __init__(self):
        super().__init__()
```

## Multiple Inheritance
```python
class Child(Parent1, Parent2):
    def __init__(self):
        Parent1.__init__(self)
        Parent2.__init__(self)
```

## Multi-level Inheritance
```python
class GrandChild(Child(Parent)):
    def __init__(self):
        super().__init__()
```

## Hierarchical Inheritance
```python
class Child1(Parent):
class Child2(Parent):
```

## Check MRO
```python
print(Class.__mro__)  # Method Resolution Order
```

## Diamond Problem Solution
Python automatically handles with C3 Linearization algorithm
```
