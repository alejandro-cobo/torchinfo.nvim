local uv = vim.uv
local utils = require("torchinfo.utils")

local torchinfo = {
    _config = {
        focus_win = false,
        gpu = -1
    }
}

function torchinfo.get_info(file_path)
    local script_path = utils.python_script_path()
    local output = {}
    local stdout = uv.new_pipe(false)
    local stderr = uv.new_pipe(false)
    local handle
    handle, _ = uv.spawn("python", {
        args = {script_path, file_path, "--gpu", torchinfo._config.gpu},
        stdio = {nil, stdout, stderr}
    },
    vim.schedule_wrap(function()
        stdout:read_stop()
        stderr:read_stop()
        stdout:close()
        stderr:close()
        handle:close()
        local buf = utils.create_window(#output, torchinfo._config.focus_win)
        vim.api.nvim_buf_call(buf, function()
            vim.api.nvim_put(output, "", false, false)
        end)
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
            print("torchinfo error:", data)
        end
    end)
    uv.run("once")
end

function torchinfo.setup(opts)
    opts = opts or {}
    torchinfo._config.focus_win = opts.focus_win
    torchinfo._config.gpu = opts.gpu or -1
end

return torchinfo

