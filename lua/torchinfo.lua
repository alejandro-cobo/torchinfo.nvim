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


local function create_window(num_lines)
    local stats = vim.api.nvim_list_uis()[1]
    local width = stats.width;
    local height = stats.height;

    local buf = vim.api.nvim_create_buf(false, true)
    vim.api.nvim_open_win(buf, true, {
        relative="editor",
        width = width,
        height = num_lines,
        col = 1,
        row = height - num_lines,
    })
    return buf
end

local function process_output(string)
    local lines = {}
    local line_number = 1
    for line in string:gmatch("([^\n]+)") do
        if line_number > 1 then
            table.insert(lines, line)
        end
        line_number = line_number + 1
    end
    return lines
end

local function get_info(file_path)
    local script_path = python_script_path()
    local ret = vim.api.nvim_exec2("!python " .. script_path .. " " .. file_path, {output=true})
    local lines = process_output(ret["output"])
    local buf = create_window(#lines)
    vim.api.nvim_put(lines, "", false, false)
    vim.api.nvim_buf_set_option(buf, "modifiable", false)
end

return {
    get_info = get_info
}

