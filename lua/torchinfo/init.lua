local uv = vim.uv
local utils = require("torchinfo.utils")

local M = {}

local config = {
    focus_win = false,
    gpu = -1,
    detailed = false
}

function M.get_info(file_path)
    local script_path = utils.python_script_path()

    local args = {script_path, file_path, "--gpu", config.gpu}
    if config.detailed then
        table.insert(args, "--detailed")
    end

    local output = {}
    local error = {}
    local stdout = uv.new_pipe(false)
    local stderr = uv.new_pipe(false)
    local handle
    handle, _ = uv.spawn("python", {
        args = args,
        stdio = {nil, stdout, stderr}
    },
        vim.schedule_wrap(function(code, _)
            stdout:read_stop()
            stderr:read_stop()
            stdout:close()
            stderr:close()
            handle:close()

            if code == 0 then
                utils.create_window(output, "Output", config.focus_win)
            else
                utils.create_window(error, "Error", true)
            end
        end)
    )

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
            for line in data:gmatch("([^\n]+)") do
                table.insert(error, line)
            end
        end
    end)
    uv.run("once")
end

function M.setup(opts)
    opts = opts or {}
    config.focus_win = opts.focus_win or config.focus_win
    config.gpu = opts.gpu or config.gpu
    config.detailed = opts.detailed or config.detailed
end

return M
