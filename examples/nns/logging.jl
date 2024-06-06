using TensorBoardLogger
import TensorBoardLogger: TBLogger
using Logging


abstract type TrainingLogger end

struct TrainingTBLogger <: TrainingLogger
    logger::TBLogger
end

struct NullLogger <: TrainingLogger end

function log_scalar!(logger, name, step, value; kwargs...) end
function log_hyperparameters!(logger, config, metrics; kwargs...) end

function log_scalar!(logger::TrainingTBLogger, name, step, value; kwargs...)
    TensorBoardLogger.log_value(logger.logger, name, value; step=step)
end
function log_hyperparameters!(logger::TrainingTBLogger, config, metrics, kwargs...)
    TensorBoardLogger.write_hparams!(logger.logger, config, metrics)
end
