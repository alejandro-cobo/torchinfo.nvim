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

function M.python_script_path()
    local path = debug.getinfo(2, "S").source:sub(2)
    if is_win() then
        path = path:gsub("/", "\\")
    end
    local sep = get_path_separator()
    path = path:match("(.*)" .. sep .. ".*" .. sep .. ".*" .. sep)
    return path .. sep .. "python" .. sep .. "get_module_info.py"
end

function M.create_window(num_lines, enter)
    local stats = vim.api.nvim_list_uis()[1]
    local width = stats.width;
    local height = stats.height;

    if num_lines > height then
        num_lines = height
    end

    local buf = vim.api.nvim_create_buf(false, true)
    vim.api.nvim_open_win(buf, enter, {
        relative = "editor",
        width = width,
        height = num_lines,
        col = 1,
        row = height - num_lines,
    })
    return buf
end

return M

