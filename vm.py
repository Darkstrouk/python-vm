import builtins
import dis
import types
import typing as tp
import operator
from itertools import chain


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.6/Include/frameobject.h#L17

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """
    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: tp.Dict[str, tp.Any],
                 frame_globals: tp.Dict[str, tp.Any],
                 frame_locals: tp.Dict[str, tp.Any]) -> None:
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value = None

        self.instructions = list(dis.get_instructions(self.code))
        self.cur_inst = 0
        self.jump_targets = {inst.offset: i for i, inst in enumerate(self.instructions) if inst.is_jump_target}

    # for  *_APPEND commands
    def peek(self, n: int) -> tp.Any:
        return self.data_stack[-n]

    def jump(self, offset: int) -> None:
        self.cur_inst = self.jump_targets[offset] - 1

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:

        while (self.cur_inst < len(self.instructions)):
            instruction = self.instructions[self.cur_inst]

            if 'INPLACE' in instruction.opname:
                self.inplaceOperator(instruction.opname.split('_', 1)[-1])
            elif 'BINARY' in instruction.opname:
                self.binaryOperator(instruction.opname.split('_', 1)[-1])
            elif 'UNARY' in instruction.opname:
                self.unaryOperator(instruction.opname.split('_', 1)[-1])
            else:
                getattr(self, instruction.opname.lower() + "_op")(instruction.argval)

            self.cur_inst += 1

        return self.return_value

    def call_function_op(self, arg: int) -> None:
        arguments = self.popn(arg)
        f = self.pop()
        self.push(f(*arguments))

    def call_function_kw_op(self, arg: int) -> None:
        keys = self.pop()
        pos_count = arg - len(keys)

        values = self.popn(len(keys))
        args = self.popn(pos_count)
        f = self.pop()
        self.push(f(*args, **dict(zip(keys, values))))

    def call_function_ex_op(self, arg: int) -> None:
        if arg & 0x01:
            kwargs = self.pop()

        args = self.pop()
        f = self.pop()

        if arg & 0x01:
            self.push(f(*args, **kwargs))
        else:
            self.push(f(*args))

    def load_name_op(self, arg: str) -> None:
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError(f"Name {arg} is not defined")

    def load_fast_op(self, arg: str) -> None:
        if arg in self.locals:
            self.push(self.locals[arg])
        else:
            raise UnboundLocalError(
                f"local variable {arg} referenced before assignment"
            )

    def load_global_op(self, arg: str) -> None:
        if arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError(f"Name {arg} is not defined")

    def load_const_op(self, arg: tp.Any) -> None:
        self.push(arg)

    def load_attr_op(self, arg: str) -> None:
        self.push(getattr(self.pop(), arg))

    def return_value_op(self, arg: tp.Any) -> None:
        self.return_value = self.pop()

    def pop_top_op(self, arg: tp.Any) -> None:
        self.pop()

    def make_function_op(self, arg: int) -> None:
        CO_VARARGS = 4
        CO_VARKEYWORDS = 8

        ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
        ERR_TOO_MANY_KW_ARGS = 'Too many keyword arguments'
        ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
        ERR_MISSING_POS_ARGS = 'Missing positional arguments'
        ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
        ERR_POSONLY_PASSED_AS_KW = 'Positional-only argument passed as keyword argument'

        name = self.pop()
        code = self.pop()

        if arg & 0x08:
            self.pop()

        if arg & 0x04:
            self.pop()

        if arg & 0x02:
            kw_defaults = self.pop()
        else:
            kw_defaults = {}

        if arg & 0x01:
            defaults = self.pop()
        else:
            defaults = {}

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:

            parsed_args: tp.Dict[str, tp.Any] = {}

            args_name = None
            if code.co_flags & CO_VARARGS:
                args_name = code.co_varnames[code.co_argcount + code.co_kwonlyargcount]
            kwargs_name = None
            if code.co_flags & CO_VARKEYWORDS:
                kwargs_name = code.co_varnames[code.co_argcount + code.co_kwonlyargcount + bool(args_name)]

            varnames_raw = list(code.co_varnames[:code.co_argcount + code.co_kwonlyargcount])
            varnames_posonly = varnames_raw[:code.co_posonlyargcount]
            varnames = varnames_raw[code.co_posonlyargcount:len(varnames_raw) - code.co_kwonlyargcount]
            kw_varnames = varnames_raw[len(varnames_raw) - code.co_kwonlyargcount:]

            arg_count = 0

            for name in varnames_posonly:
                if name in kwargs and kwargs_name is None:
                    raise TypeError(ERR_POSONLY_PASSED_AS_KW)

                if arg_count == len(args):
                    if name in defaults:
                        parsed_args[name] = defaults[name]
                    else:
                        raise TypeError(ERR_MISSING_POS_ARGS)
                else:
                    parsed_args[name] = args[arg_count]
                    arg_count += 1

            from_kw = False

            for name in varnames:
                if name in kwargs:
                    parsed_args[name] = kwargs[name]
                    from_kw = True
                    continue

                if arg_count == len(args):
                    if name in defaults:
                        parsed_args[name] = defaults[name]
                    else:
                        raise TypeError(ERR_MISSING_POS_ARGS)
                else:
                    parsed_args[name] = args[arg_count]
                    arg_count += 1

            if args_name is not None:
                parsed_args[args_name] = tuple(args[arg_count:])

            if arg_count != len(args) and args_name is None:
                if from_kw:
                    raise TypeError(ERR_MULT_VALUES_FOR_ARG)
                else:
                    raise TypeError(ERR_TOO_MANY_POS_ARGS)

            for name in kw_varnames:
                if name in kwargs:
                    parsed_args[name] = kwargs[name]

                elif kw_defaults is not None and name in kw_defaults:
                    parsed_args[name] = kw_defaults[name]
                else:
                    raise TypeError(ERR_MISSING_KWONLY_ARGS)

            not_used = {key: val for key, val in kwargs.items() if key not in parsed_args or key in varnames_posonly}

            if kwargs_name is not None:
                parsed_args[kwargs_name] = not_used
            elif len(not_used):
                raise TypeError(ERR_TOO_MANY_KW_ARGS)

            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)
            return frame.run()

        self.push(f)

    def dup_top_op(self, arg: tp.Any) -> None:
        self.push(self.top())

    def dup_top_two_op(self, arg: tp.Any) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1, tos, tos1, tos)

    def map_add_op(self, arg: int) -> None:
        key, value = self.popn(2)
        map_ = self.peek(arg)
        map_[key] = value

    def set_add_op(self, arg: int) -> None:
        value = self.pop()
        set_ = self.peek(arg)
        set_.add(value)

    def nop_op(self, arg: tp.Any) -> None: pass

    def pop_block_op(self, arg: tp.Any) -> None:
        raise NotImplementedError('POP_BLOCK')

    def pop_except_op(self, arg: tp.Any) -> None:
        raise NotImplementedError('POP_EXCEPT')

    def rot_two_op(self, arg: tp.Any) -> None:
        *rest, tos = self.popn(2)
        self.push(tos, *rest)

    def rot_three_op(self, arg: tp.Any) -> None:
        *rest, tos = self.popn(3)
        self.push(tos, *rest)

    def rot_four_op(self, arg: tp.Any) -> None:
        *rest, tos = self.popn(4)
        self.push(tos, *rest)

    def store_name_op(self, arg: str) -> None:
        const = self.pop()
        self.locals[arg] = const

    def store_global_op(self, arg: str) -> None:
        const = self.pop()
        self.globals[arg] = const

    def store_fast_op(self, arg: str) -> None:
        self.locals[arg] = self.pop()

    def store_attr_op(self, arg: str) -> None:
        value, object_ = self.popn(2)
        setattr(object_, arg, value)

    def inplaceOperator(self, opname: str) -> None:
        x, y = self.popn(2)
        if opname == 'POWER':
            x **= y
        elif opname == 'MULTIPLY':
            x *= y
        elif opname in ['DIVIDE', 'FLOOR_DIVIDE']:
            x //= y
        elif opname == 'TRUE_DIVIDE':
            x /= y
        elif opname == 'MODULO':
            x %= y
        elif opname == 'ADD':
            x += y
        elif opname == 'SUBTRACT':
            x -= y
        elif opname == 'LSHIFT':
            x <<= y
        elif opname == 'RSHIFT':
            x >>= y
        elif opname == 'AND':
            x &= y
        elif opname == 'XOR':
            x ^= y
        elif opname == 'OR':
            x |= y
        else:
            raise ValueError

        self.push(x)

    BINARY_OPERATORS = {
        'POWER':    pow,
        'MULTIPLY': operator.mul,
        'DIVIDE':   getattr(operator, 'div', lambda x, y: None),
        'FLOOR_DIVIDE': operator.floordiv,
        'TRUE_DIVIDE':  operator.truediv,
        'MODULO':   operator.mod,
        'ADD':      operator.add,
        'SUBTRACT': operator.sub,
        'SUBSCR':   operator.getitem,
        'LSHIFT':   operator.lshift,
        'RSHIFT':   operator.rshift,
        'AND':      operator.and_,
        'XOR':      operator.xor,
        'OR':       operator.or_,
    }

    def binaryOperator(self, opname: str) -> None:
        x, y = self.popn(2)
        self.push(self.BINARY_OPERATORS[opname](x, y))

    UNARY_OPERATORS = {
        'POSITIVE': operator.pos,
        'NEGATIVE': operator.neg,
        'NOT':      operator.not_,
        'CONVERT':  repr,
        'INVERT':   operator.invert,
    }

    def unaryOperator(self, opname: str) -> None:
        x = self.pop()
        self.push(self.UNARY_OPERATORS[opname](x))

    def unpack_sequence_op(self, arg: int) -> None:
        sequence = self.pop()
        for item in reversed(sequence):
            self.push(item)

    def compare_op_op(self, opname: str) -> None:
        ops = {
            '<': operator.lt,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne,
            '>': operator.gt,
            '>=': operator.ge,
            'in': lambda x, y: x in y,
            'not in': lambda x, y: x not in y,
            'is': lambda x, y: x is y,
            'is not': lambda x, y: x is not y,
            'exception match': lambda x, y: issubclass(x, Exception) and issubclass(x, y),
        }

        x, y = self.popn(2)
        self.push(ops[opname](x, y))

    def list_append_op(self, arg: int) -> None:
        val = self.pop()
        list_ = self.peek(arg)
        list_.append(val)

    # JUMPS

    def jump_forward_op(self, offset: int) -> None:
        self.jump(offset)

    def jump_absolute_op(self, offset: int) -> None:
        self.jump(offset)

    def pop_jump_if_true_op(self, offset: int) -> None:
        tos = self.pop()
        if tos:
            self.jump(offset)

    def pop_jump_if_false_op(self, offset: int) -> None:
        tos = self.pop()
        if not tos:
            self.jump(offset)

    def jump_if_true_or_pop_op(self, offset: int) -> None:
        tos = self.top()
        if tos:
            self.jump(offset)
        else:
            self.pop()

    def jump_if_false_or_pop_op(self, offset: int) -> None:
        tos = self.top()
        if not tos:
            self.jump(offset)
        else:
            self.pop()

    def extended_arg_op(self, arg: tp.Any) -> None: pass

    def get_iter_op(self, arg: tp.Any) -> None:
        self.push(iter(self.pop()))

    def for_iter_op(self, offset: int) -> None:
        iterobj = self.top()
        try:
            next_iterobj = next(iterobj)
            self.push(next_iterobj)
        except StopIteration:
            self.pop()
            self.jump(offset)

    def print_expr_op(self, arg: tp.Any) -> None:
        print(self.pop())

    # BUILDS

    def build_tuple_op(self, arg: int) -> None:
        seq = self.popn(arg)
        self.push(tuple(seq))

    def build_tuple_unpack(self, arg: int) -> None:
        seq = self.popn(arg)
        self.push(tuple(chain(*seq)))

    def build_tuple_unpack_with_call_op(self, arg: int) -> None:
        self.build_tuple_unpack(arg)

    def build_list_op(self, arg: int) -> None:
        seq = self.popn(arg)
        self.push(seq)

    def build_list_unpack_op(self, arg: int) -> None:
        seq = self.popn(arg)
        self.push(list(chain(*seq)))

    def build_set_op(self, arg: int) -> None:
        seq = self.popn(arg)
        self.push(set(seq))

    def build_set_unpack_op(self, arg: int) -> None:
        seq = self.popn(arg)
        self.push(set(chain(*seq)))

    def build_map(self, arg: int) -> None:
        dic = {}
        seq = self.popn(2 * arg)

        for i in range(0, 2 * arg, 2):
            dic[seq[i]] = seq[i - 1]

        self.push(dic)

    def build_map_unpack_op(self, arg: int) -> None:
        if arg == 0:
            self.push({})
            return

        seq = self.popn(arg)
        dic = {}
        for d in seq:
            dic.update(d)

        self.push(dic)

    def build_map_unpack_with_call_op(self, arg: int) -> None:
        self.build_map_unpack_op(arg)

    def build_const_key_map_op(self, arg: int) -> None:
        keys = self.pop()
        vals = self.popn(arg)

        self.push(dict(zip(keys, vals)))

    def build_slice_op(self, arg: int) -> None:
        seq = self.popn(arg)
        self.push(slice(*seq))

    def build_string_op(self, arg: int) -> None:
        seq = self.popn(arg)
        self.push(''.join(seq))

    def store_subscr_op(self, arg: tp.Any) -> None:
        value, object_, subscr = self.popn(3)
        object_[subscr] = value

    # DELS

    def delete_subscr_op(self, arg: tp.Any) -> None:
        object_, subscr = self.popn(2)
        del object_[subscr]

    def delete_name_op(self, arg: str) -> None:
        del self.locals[arg]

    def delete_global_op(self, arg: str) -> None:
        del self.globals[arg]

    def delete_fast_op(self, arg: str) -> None:
        del self.locals[arg]

    def delete_attr_op(self, arg: str) -> None:
        delattr(self.pop(), arg)

    # IMPORTS

    def import_from_op(self, arg: str) -> None:
        module = self.top()
        self.push(getattr(module, arg))

    def import_name_op(self, arg: str) -> None:
        level, fromlist = self.popn(2)
        self.push(
            __import__(arg, self.globals, self.locals, fromlist, level)
        )

    def import_star_op(self, arg: tp.Any) -> None:
        module = self.pop()
        for attr in dir(module):
            if attr[0] != '_':
                self.locals[attr] = getattr(module, attr)


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_text_or_obj: code for interpreting
        """
        globals_context: tp.Dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()
