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

function M.create_window(lines, title, enter)
    local num_lines = #lines
    if num_lines == 0 then
        return
    end

    local width = vim.o.columns
    local height = vim.o.lines
    if num_lines > height then
        num_lines = height
    end

    local buf = vim.api.nvim_create_buf(false, true)
    local win = vim.api.nvim_open_win(buf, enter, {
        relative = "editor",
        style = "minimal",
        border = "rounded",
        title = title,
        width = width,
        height = num_lines,
        col = 1,
        row = height - num_lines,
    })
    vim.keymap.set("n", "q", function()
        vim.api.nvim_win_close(win, true)
    end, { buffer = buf })

    vim.api.nvim_buf_set_lines(buf, 0, -1, true, lines)
    vim.api.nvim_set_option_value("modifiable", false, { buf = buf })
    return buf
end

return M
