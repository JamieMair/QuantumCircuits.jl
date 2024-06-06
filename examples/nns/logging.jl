using TensorBoardLogger
import TensorBoardLogger: TBLogger

abstract type TrainingLogger end

struct TrainingTBLogger <: TrainingLogger
    logger::TBLogger
end

struct NullLogger <: TrainingLogger end

function log_scalar!(logger, name, value; kwargs...) end
function log_hyperparameters!(logger, config; kwargs...) end

function log_scalar!(logger::TrainingTBLogger, name, value; kwargs...)
    TensorBoardLogger.log_value(logger, name, value)
end
function log_hyperparameters!(logger::TrainingTBLogger, config::AbstractDict; metrics::AbstractArray{String}, kwargs...)
    TensorBoardLogger.write_hparams!(logger, config, metrics)
end
