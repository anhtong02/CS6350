## How to use primal SVM:
initalize: svm = PrimalSVM(C, T, schedule_fn, params)
 1. C is your tradeoff
 2. T is epoch
 3. schedule_fn is your function to compute learning rate
 4. Params for your shedule function, should be a dict, the algorithm will unpack this for you.

Then train : svm.train(X_train, y_train)
 1. X_train is your training data X
 2. y_train is your training data label y

 Get errors:
 1. train_error = svm.error(X_train, y_train)
 2. test_error = svm.error(X_test, y_test)

## How to use Dual: 
-initialize: alpha = `train_dual_svm(X_train, y_train, C)`

-retract weights and bias by: `w, b = get_w_b(X_train, y_train, alpha, C)`

-get errors: 
1. `train_error_dual = dual_error(X_train, y_train, w, b)`
2. `test_error_dual = dual_error(X_test, y_test, w, b)`

## How to use Dual with Kernel:
- Initialize: `alpha = train_dual_svm_kernel(X_train, y_train, C, gaussian_kernel, g)`
 1. g is your gamma value
 
