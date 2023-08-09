import multiprocessing

import testing_util as test_util
import numpy as np

"""
Running set in a separate process
https://github.com/hendrycks/apps/blob/83d925041b1c43c32b56d444bb315f729f4ff633/eval/test_one_solution.py#L57
"""


def _temp_run(prob_path, output_str, mode, public_test_cases, result, error_queue):
    try:
        result.append(
            test_util.run_test(prob_path=prob_path, test=output_str, mode=mode, public_test_cases=public_test_cases))
    except Exception as e:
        error_queue.put(e)


def check_correctness(prob_path, output_str, mode, public_test_cases):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    # python自带的多进程管理器，可以保证多进程安全
    manager = multiprocessing.Manager()
    # manager创建了一个共享的数据结构，用于进程通信，这是并发安全的
    result = manager.list()
    error_queue = multiprocessing.Queue()
    # 创建子进程, target是要执行的函数，args是待执行函数所需要的参数
    p = multiprocessing.Process(target=_temp_run, args=(prob_path, output_str, mode, public_test_cases, result, error_queue))
    p.start()
    p.join(timeout=10)
    # 这里可以捕获异常，加入reflexion的方法
    reflexion_error = "None error"
    if not error_queue:
        reflexion_error += error_queue.get(block=False)
    if p.is_alive():
        p.kill()
    if not result:
        # Reamark: ideally we would consider that all tests failed but we can't access number of tests here easily
        # so we use 21=the average number of tests for a smaple in the test split instead
        # result为空，说明测试全部失败了，具体数量拿不到，取了个均值。为什么拿不到具体数量？数量不就是测试用例的总数量，
        avg_number_tests = 21
        # 21搁-1
        result = [[-1] * avg_number_tests]
    return result[0], reflexion_error


def compute_reward(prob_path, output_str, mode = 'train', public_test_cases = None, return_info = False):
    """
    A utility function that computes the reward given problem path and output string of our model
    It is rewarded by the number of tests passed. When passing the same number of tests.
    """
    # from https://github.com/hendrycks/apps/blob/83d925041b1c43c32b56d444bb315f729f4ff633/eval/test_one_solution.py#L141
    reflexion_error = "None error"
    try:
        # 检查正确性，通过直接运行代码
        curr_res, reflexion_error = check_correctness(prob_path, output_str, mode, public_test_cases)
        fixed = []
        # 转换一下格式
        for e in curr_res:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            fixed.append(e)

        curr_res = fixed
        # if not np.all(curr_res):
        #     print(f"Results were not all True: {curr_res}")
    except Exception as e:
        print(f"test framework exception = {repr(e)}{e}\n")
        curr_res = []

    # How to read results [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case")
    assert isinstance(curr_res, list)
    # 计算测试用例的通过率
    pass_rate = np.mean(np.asarray(curr_res) > 0) if len(curr_res) > 0 else 0
    # 计算编译错误和运行时错误的比例
    if return_info:
        info = {"compile_error": curr_res.count(-2) / len(curr_res),
                "runtime_error": curr_res.count(-1) / len(curr_res)}
        return pass_rate, info, reflexion_error
    else:
        return pass_rate, reflexion_error


def get_program_quality(s):
    """
    For now, only consider the length of the program. The shorter, the better.
    """
    return np.exp(- len(s) / 20)


if __name__ == '__main__':
    # 测试
    func = "def add(a, b):\n    while True:\n        x = 1\n    return a + b"
    print(func)

