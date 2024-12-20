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

local function get_info(file_path)
    local script_path = python_script_path()
    local output = {}
    local uv = vim.uv
    local stdout = uv.new_pipe(false)
    local stderr = uv.new_pipe(false)
    local handle
    handle, _ = uv.spawn("python", {
        args = {script_path, file_path},
        stdio = { nil, stdout, stderr }
    },
    vim.schedule_wrap(function()
        stdout:read_stop()
        stderr:read_stop()
        stdout:close()
        stderr:close()
        handle:close()
        local buf = create_window(#output)
        vim.api.nvim_put(output, "", false, false)
        vim.api.nvim_buf_set_option(buf, "modifiable", false)
    end))

    uv.read_start(stdout, function(err, data)
        assert(not err, err)
        if data then
            for line in data:gmatch("([^\n]+)") do
                table.insert(output, line)
            end
        end
    end)

    uv.read_start(stderr, function(err, data)
        assert(not err, err)
        if data then
            print("torchinfo error", data)
        end
    end)
    uv.run("once")
end

return {
    get_info = get_info
}

