local M = {}

local function is_win()
  return package.config:sub(1, 1) == "\\"
end

local function get_path_separator()
  if is_win() then
    return "\\"
  end
  return "/"
end

local function python_script_path()
  local str = debug.getinfo(2, "S").source:sub(2)
  if is_win() then
    str = str:gsub("/", "\\")
  end
  local sep = get_path_separator()
  str = str:match("(.*)" .. sep)
  str = str:match("(.*)" .. sep)
  return str .. sep .. "python" .. sep .. "get_module_info.py"
end

M.get_info = function(file_path)
    local script_path = python_script_path()
    local ret = vim.api.nvim_exec2("!python " .. script_path .. " " .. file_path, {output=true})
    local ret_stdout = ret["output"]
    ret_stdout = ret_stdout:sub(ret_stdout:find("\n"), -1)
    vim.api.nvim_echo({{ret_stdout}}, true, {})
end

return M

