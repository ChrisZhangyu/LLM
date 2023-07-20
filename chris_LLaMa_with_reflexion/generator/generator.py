from typing import Optional, Union, List


class Generator:
    """
        prompt_list保存了各种prompt的语句
    """
    def __init__(self, prompt_list):
        self.prompt_list = prompt_list

    def gen_reflection(
            self,
            func: str,
            feedback: str,
            model,
    ) -> str:
        func = f'Function implementation:\n{func}\n\n' \
              f'Unit test results:\n{feedback}' \
              f'Self-reflection:'
        self_reflection_few_shot = self.prompt_list["few_shot"]
        if not self_reflection_few_shot:
            func = self_reflection_few_shot + '\n\n' + func
        request_message = self.prompt_list["completion_instruction"] + "\n" + func
        reflection = model(request_message)
        return reflection

    """
        根据提示生成代码，后续需要调整模型以便应用code_count和temperature参数
    """
    def get_code(self,
                 func_signature,
                 model,
                 prev_code,
                 feedback,
                 self_reflection,
                 temperature,
                 code_count=1,
                 ) -> Union[str, List[str]]:
        request_message = f"{self.prompt_list['reflexion_few_shot']}\n" \
                          f"[previous impl]:\n{prev_code}\n\n" \
                          f"[unit test results from previous " \
                          f"impl]:\n{feedback}\n\n" \
                          f"[reflection on previous impl]:\n{self_reflection}\n\n[improve" \
                          f"d impl]:\n{func_signature}"
        code = model(request_message)
        if code_count == 1:
            return code



